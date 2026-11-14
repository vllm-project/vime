import logging

import ray

from vime.platforms import current_platform

if current_platform().is_npu:
    import vime.backends.megatron_utils  # noqa: F401

from vime.backends.speculative_training.draft_group import DraftTrainingLoop
from vime.data.checkpoint import save_checkpoint
from vime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
from vime.ray.placement_group import create_draft_model, create_rollout_manager, create_training_models
from vime.utils.arguments import parse_args
from vime.utils.cleanup import Cleanup
from vime.utils.misc import should_run_periodic_action


def train(args, pgs, rollout_manager, num_rollout_per_epoch, restore_plan):
    release_train = args.release_train

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    draft_model = create_draft_model(args, actor_model)
    draft_loop = DraftTrainingLoop(args, draft_model) if draft_model is not None else None

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
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

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        actor_train_results = None
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                actor_train_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs)
                )
            else:
                ray.get(value_refs)
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        # Runtime completion releases queue capacity, without claiming that the
        # model/optimizer or this data progress have a durable joint checkpoint.
        ray.get(rollout_manager.training_completed.remote(rollout_id))

        # Draft stages: train.py only sequences them; interval decisions and
        # snapshot bookkeeping live in DraftTrainingLoop.
        draft_snapshot = None
        if draft_loop is not None and actor_trains and actor_train_results is not None:
            draft_loop.collect(rollout_id, actor_train_results)
            if draft_loop.should_train(rollout_id):
                draft_loop.train(rollout_id)
            if draft_loop.should_publish(rollout_id):
                draft_loop.prepare_snapshot()
            draft_snapshot = draft_loop.pending_snapshot

        actor_save_due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        if release_train or actor_save_due:
            save_checkpoint(
                args,
                rollout_id,
                actor_model,
                critic_model,
                rollout_manager,
                actor_trains=actor_trains,
                restore_plan=restore_plan,
            )

        if draft_loop is not None and draft_loop.should_save(rollout_id, actor_save_due):
            draft_loop.save(rollout_id)

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        if draft_snapshot is not None:
            actor_model.set_external_draft_weights(*draft_snapshot)
        was_paused = ray.get(rollout_manager.pause_rollout_admission.remote())
        actor_model.update_weights()
        if draft_snapshot is not None:
            draft_loop.commit_publish(rollout_id)
        # Final evaluation uses the synchronized engines directly. Keep training
        # producers paused when no later rollout will consume their new work.
        if rollout_id + 1 < args.num_rollout:
            ray.get(rollout_manager.resume_rollout_admission.remote(was_paused))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    if draft_loop is not None:
        draft_loop.release()


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
