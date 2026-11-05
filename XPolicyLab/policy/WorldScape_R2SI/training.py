"""Reserved WSP training interface; inference does not depend on a trainer."""


class WSPTrainingUnavailable(NotImplementedError):
    """The training backend has not been connected in this release."""


def require_training_backend():
    raise WSPTrainingUnavailable(
        "WSP training backend is not connected (reserved interface / stub). "
        "No training job or checkpoint was created. WAM inference remains available."
    )


def train_wsp(*, python, xpolicy_root, config, snapshot, gpus, max_steps,
              timeout_s, overrides=()):
    """Reserve the native trainer's call signature without starting any work.

    A future backend must return a verified job/checkpoint result. This stub
    raises before loading data, checking GPUs, spawning a process or writing
    outputs, so callers cannot mistake an unimplemented run for completed training.
    """
    require_training_backend()
