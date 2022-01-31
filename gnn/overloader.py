
from functools import wraps
def overload(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) +  len(kargs) == 2:
            # for inputs like model(g)
            if len(args) == 2:
                g = args[1]
            # for inputs like model(graph=g)
            else:
                g = kargs['graph']
            return func(args[0], g.x, g.edge_index, g.edge_attr, g.batch)
        elif len(args) +  len(kargs) == 5:
            # for inputs like model(x, ..., batch)
            if len(args) == 5:
                return func(*args)
            # for inputs like model(x=x, ..., batch=batch)
            else:
                return func(args[0], **kargs)
        elif len(args) +  len(kargs) == 6:
            # for inputs like model(x, ..., batch, pos)
            if len(args) == 6:
                return func(*args[:-1])
            # for inputs like model(x=x, ..., batch=batch, pos=pos)
            else:
                return func(args[0], kargs['x'], kargs['edge_index'], kargs['edge_attr'], kargs['batch'])
        else:
            raise TypeError
    return wrapper