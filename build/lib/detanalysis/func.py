# Functions to te used as decorators for cut/feature scripts

__all__= ['version', 'authors', 'date', 'description', 'contact']


# cut version
def version(par):
    def _wrapper(func):
        func.version = float(par)
        return func
    return _wrapper

# author(s)
def authors(par):
    def _wrapper(func):
        func.authors = str(par)
        return func
    return _wrapper

# (brief) description
def description(par):
    def _wrapper(func):
        func.description = str(par)
        return func
    return _wrapper

# contact 
def contact(par):
    def _wrapper(func):
        func.contact = str(par)
        return func
    return _wrapper

# date 
def date(par):
    def _wrapper(func):
        func.date = str(par)
        return func
    return _wrapper




