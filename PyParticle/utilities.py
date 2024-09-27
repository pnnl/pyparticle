#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some basic functions used by the other modules

@author: Laura Fierce
"""


def get_number(string_val):
    if string_val.endswith('\n'):
        string_val = string_val[:-2]
        
    if '×' in string_val:
        idx = string_val.find('×')
        front_part = float(string_val[:idx])
        back_part = string_val[(idx+1):][2:]
        if back_part.startswith('−'):
            exponent = -float(back_part[1:])
        else:
            exponent = float(back_part)
        number = front_part*10.**exponent
    else:
        number = float(string_val)
    return number


import sys
import pickle
import importlib
import io
import traceback
import subprocess

class Py3Wrapper(object):
    def __init__(self, mod_name, func_name):
        self.mod_name = mod_name
        self.func_name = func_name

    def __call__(self, *args, **kwargs):
        p = subprocess.Popen(['python3', '-m', 'py3bridge',
                              self.mod_name, self.func_name],
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE)
        stdout, _ = p.communicate(pickle.dumps((args, kwargs)))
        data = pickle.loads(stdout)
        if data['success']:
            return data['result']
        else:
            raise Exception(data['stacktrace'])

def main():
    try:
        target_module = sys.argv[1]
        target_function = sys.argv[2]
        args, kwargs = pickle.load(sys.stdin.buffer)
        mod = importlib.import_module(target_module)
        func = getattr(mod, target_function)
        result = func(*args, **kwargs)
        data = dict(success=True, result=result)
    except Exception:
        st = io.StringIO()
        traceback.print_exc(file=st)
        data = dict(success=False, stacktrace=st.getvalue())

    pickle.dump(data, sys.stdout.buffer, 2)

if __name__ == '__main__':
    main()