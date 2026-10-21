import logging

import ray

from vime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.misc import should_run_periodic_action


def train_queue_batch(args, actor_model, rollout_manager, rollout_id, rollout_data_ref):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    coordinator = ray.get(rollout_manager.queue_training_boundary.remote(), timeout=args.transfer_queue_timeout_s)
    stopped = Event()

    def heartbeat():
        while not stopped.wait(args.transfer_queue_lease_s / 3):
            ray.get(coordinator.heartbeat.remote(), timeout=args.transfer_queue_timeout_s)

    completed = False
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending_heartbeat = executor.submit(heartbeat)
        try:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref), timeout=args.transfer_queue_timeout_s)
            stopped.set()
            pending_heartbeat.result()
            ray.get(rollout_manager.queue_training_complete.remote(rollout_id), timeout=args.transfer_queue_timeout_s)
            completed = True
        finally:
            stopped.set()
            if not completed:
                try:
                    ray.get(coordinator.fail.remote(), timeout=args.transfer_queue_timeout_s)
                except (TimeoutError, ray.exceptions.RayError):
                    logging.getLogger(__name__).warning("Queue fail-stop acknowledgment failed", exc_info=True)


def train(args):
    configure_logger()
    release_train = args.release_train
    queue_enabled = args.transfer_queue_mode != "off"
    queue_timeout = args.transfer_queue_timeout_s if queue_enabled else None

    def publish_actor():
        if queue_enabled:
            actor_model.update_weights(timeout_s=queue_timeout)
        else:
            actor_model.update_weights()

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with vLLM engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    publish_actor()

    if queue_enabled:
        ray.get(rollout_manager.queue_confirm_publication.remote(), timeout=args.transfer_queue_timeout_s)

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save_training_state(rollout_id, actor_trains, force_sync):
        if actor_trains:
            if queue_enabled:
                actor_model.save_model(rollout_id, force_sync=force_sync, timeout_s=queue_timeout)
            else:
                actor_model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic:
            critic_model.save_model(rollout_id, force_sync=force_sync)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id), timeout=queue_timeout)

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id), timeout=queue_timeout)

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        elif queue_enabled:
            train_queue_batch(args, actor_model, rollout_manager, rollout_id, rollout_data_ref)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        save_this_round = release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        force_sync = release_train or rollout_id == args.num_rollout - 1
        if save_this_round and not queue_enabled:
            save_training_state(rollout_id, actor_trains, force_sync)

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        publish_actor()
        if queue_enabled:
            ray.get(rollout_manager.queue_confirm_publication.remote(), timeout=args.transfer_queue_timeout_s)
            if save_this_round:
                save_training_state(rollout_id, actor_trains, force_sync=True)
                ray.get(rollout_manager.queue_save.remote(rollout_id), timeout=args.transfer_queue_timeout_s)

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
