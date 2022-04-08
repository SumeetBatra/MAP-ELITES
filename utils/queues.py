import os


def get_mp_queue(buffer_size_bytes=1_000_000):
    from faster_fifo import Queue as MpQueue
    # noinspection PyUnresolvedReferences
    import faster_fifo_reduction

    return MpQueue(buffer_size_bytes)