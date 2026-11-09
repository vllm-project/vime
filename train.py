import logging

import ray

from vime.data.checkpoint import save_checkpoint
from vime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from vime.ray.placement_group import create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.cleanup import Cleanup
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


def train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
    release_train = args.release_train
    queue_enabled = getattr(args, "transfer_queue_mode", "off") != "off"
    queue_timeout = args.transfer_queue_timeout_s if queue_enabled else None

    def publish_actor():
        if queue_enabled:
            actor_model.update_weights(timeout_s=queue_timeout)
        else:
            actor_model.update_weights()

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    publish_actor()
    if queue_enabled:
        ray.get(rollout_manager.queue_confirm_publication.remote(), timeout=args.transfer_queue_timeout_s)
    # Reattached async producers stay paused until restored weights are installed.
    # This startup notification resumes them once, before entering the train loop.
    ray.get(rollout_manager.training_ready.remote())

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

    def save_training_state(rollout_id, actor_trains):
        save_checkpoint(
            args,
            rollout_id,
            actor_model,
            critic_model,
            rollout_manager,
            actor_trains=actor_trains,
            restore_plan=restore_plan,
            timeout_s=queue_timeout,
        )

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

        # Runtime completion releases queue capacity, without claiming that the
        # model/optimizer or this data progress have a durable joint checkpoint.
        ray.get(rollout_manager.training_completed.remote(rollout_id))

        save_this_round = release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        if save_this_round and not queue_enabled:
            save_training_state(rollout_id, actor_trains)

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
        publish_actor()
        if queue_enabled:
            ray.get(rollout_manager.queue_confirm_publication.remote(), timeout=args.transfer_queue_timeout_s)
            if save_this_round:
                save_training_state(rollout_id, actor_trains)
                ray.get(rollout_manager.queue_save.remote(rollout_id), timeout=args.transfer_queue_timeout_s)
        # Final evaluation uses the synchronized engines directly. Keep training
        # producers paused when no later rollout will consume their new work.
        if rollout_id + 1 < args.num_rollout:
            ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))


def main(args, restore_plan):
    """Own the attempt's exit policy; actors only clean up their own resources."""
    with Cleanup() as cleanup:
        init_tracking(args)
        startup = None
        try:
            # Create or reattach rollout resources, then run this training attempt.
            startup = create_rollout_manager(args, restore_plan=restore_plan)
            train(
                startup.args, startup.placements, startup.manager, startup.num_rollout_per_epoch, startup.restore_plan
            )
        except BaseException:
            # Failures and interruptions retain serving and replay data for a manual
            # restart. Detach this driver's trainers instead of disposing the session.
            if startup is not None:
                try:
                    startup.close(failed=True)
                except Exception:
                    logging.getLogger(__name__).exception("Failed to clean up training attempt")
            raise
        else:
            startup.close(failed=False)
        finally:
            # Finish this driver's tracking on success, startup failure, or interruption.
            cleanup.run("finish driver tracking", finish_tracking, args)


if __name__ == "__main__":
    args, restore_plan = parse_args(return_restore_plan=True)
    configure_logger()
    main(args, restore_plan)
