import sys
from signal_slot import *
from logger import log


class Variation(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)

    def mutate(self, x):
        x = x + 1
        log.debug(f'[Object {self.object_id}] x is mutated to {x}')
        self.s1.emit((self.object_id, x))

    @signal
    def s1(self): pass



class Evaluation(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.f = lambda x: 2 * x

    def evaluate(self, x):
        x = self.f(x)
        log.debug(f'[Object {self.object_id}] x is evaluated to {x}')
        self.s2.emit((self.object_id, x))

    def on_reply(self, oid, x):
        log.debug(f'Received {x} from object {oid}')
        self.evaluate(x)

    @signal
    def s2(self): pass



def main():
    p1 = EventLoop('loop1')
    p2 = EventLoop('loop2')
    p = EventLoopProcess('subp', daemon=True)
    var_worker = Variation(p.event_loop, 'var')
    eval_worker = Evaluation(p.event_loop, 'eval')

    var_worker.s1.connect(eval_worker.on_reply)
    var_worker.mutate(x=2)



if __name__ == '__main__':
    sys.exit(main())