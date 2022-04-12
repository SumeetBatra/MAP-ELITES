
import sys

from signal_slot import EventLoopObject, signal, EventLoop, EventLoopProcess
from logger import log


class Variation(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)

    @signal
    def s1(self): pass

    @signal
    def done(self): pass

    def mutate(self, x):
        x = x + 1
        log.debug(f'[Object {self.object_id}] x is mutated to {x}')
        self.s1.emit(self.object_id, x)


class Evaluation(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.f = lambda x: 2 * x

    def evaluate(self, x):
        x = self.f(x)
        log.debug(f'[Object {self.object_id}] x is evaluated to {x}')
        self.s2.emit((self.object_id, x))
        self.done.emit()

    def on_reply(self, oid, x):
        log.debug(f'Received {x} from object {oid}')
        self.evaluate(x)

    @signal
    def s2(self): pass

    @signal
    def done(self): pass


def main():
    l1 = EventLoop('loop1')

    p2 = EventLoopProcess('p2')

    var_worker = Variation(l1, 'var')
    eval_worker = Evaluation(p2.event_loop, 'eval')

    var_worker.s1.connect(eval_worker.on_reply)
    var_worker.done.connect(p2.stop)
    eval_worker.done.connect(l1.stop)

    p2.start()

    var_worker.mutate(x=2)
    var_worker.done.emit()

    l1.exec()
    p2.join()


if __name__ == '__main__':
    sys.exit(main())