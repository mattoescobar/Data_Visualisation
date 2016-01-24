import numpy as np


class Computer(object):

    def __init__(self,  stack, instructions, stack_pointer=-1,
                 program_counter=0):
        self.stack_pointer = stack_pointer
        self.program_counter = program_counter
        self.stack = stack
        self.instructions = instructions
        self.instruction_mapping = {
            "ADD": self.add,
            "CALL": self.call,
            "PRN": self.prn,
            "PUSH": self.push,
            "RET": self.ret,
            "STOP": self.stop
        }
        self.running = True

    def set_address(self, address):
        self.program_counter = address
        return self

    def insert(self, instruction_name, instruction_arg=None):
        inst = np.array([self.instruction_mapping[instruction_name],
                         instruction_arg])
        self.instructions[self.program_counter] = inst
        self.program_counter += 1
        return self

    def execute(self):
        while self.running:
            instruction_pair = self.instructions[self.program_counter]
            instruction_method = instruction_pair[0]
            instruction_arg = instruction_pair[1]
            if instruction_arg is None:
                instruction_method()
            else:
                instruction_method(instruction_arg)

    def add(self):
        addend1 = self.pop()
        addend2 = self.pop()
        self.push(addend1+addend2)

    def call(self, address):
        self.program_counter = address

    def pop(self):
        arg = self.stack[self.stack_pointer]
        self.stack[self.stack_pointer] = None
        self.stack_pointer -= 1
        return arg

    def prn(self):
        top_stack_arg = self.pop()
        print top_stack_arg
        self.program_counter += 1

    def push(self, arg):
        self.stack_pointer += 1
        self.stack[self.stack_pointer] = arg
        self.program_counter += 1

    def ret(self):
        address = self.pop()
        self.program_counter = address

    def stop(self):
        self.running = False
