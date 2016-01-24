import numpy as np


class Computer(object):

    def __init__(self,  stack, stack_pointer=0, program_counter=0):
        """ Initialization of the computer simulator using a single stack
        memory

        :param stack: stack memory
        :param stack_pointer: pointer managing values in the stack
        :param program_counter: counter managing instructions in the stack
        """
        self.stack_pointer = stack_pointer
        self.program_counter = program_counter
        self.stack = stack
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
        """ Set Program Counter to this address

        :param address: address in the stack
        """
        self.program_counter = address
        if self.program_counter > self.stack_pointer:
            self.stack_pointer = self.program_counter
        return self

    def insert(self, instruction_name, instruction_arg=None):
        """ Insert instruction in the stack

        :param instruction_name: name of the instruction
        :param instruction_arg: argument of the respective instruction
        """
        instruction = np.array([self.instruction_mapping[instruction_name],
                         instruction_arg])
        self.stack[self.program_counter] = instruction
        self.program_counter += 1
        if self.program_counter > self.stack_pointer:
            self.stack_pointer = self.program_counter
        return self

    def execute(self):
        """ Execute all instructions in the stack """
        while self.running:
            instruction_pair = self.stack[self.program_counter]
            instruction_method = instruction_pair[0]
            instruction_arg = instruction_pair[1]
            if instruction_arg is None:
                instruction_method()
            else:
                instruction_method(instruction_arg)

    def add(self):
        """ Pop two upper stack values, add them together and push the result
        on top of the stack """
        addend1 = self.pop()
        addend2 = self.pop()
        self.push(addend1+addend2)

    def call(self, address):
        """ Set Program Counter to this address

        :param address: address in the stack
        """
        self.program_counter = address

    def pop(self):
        """ Pop the top stack value """
        arg = self.stack[self.stack_pointer]
        self.stack[self.stack_pointer] = None
        self.stack_pointer -= 1
        return arg

    def prn(self):
        """ Pop the top stack value and print it out """
        top_stack_value = self.pop()
        print top_stack_value
        self.program_counter += 1

    def push(self, arg):
        """ Push 'arg' to the top of the stack

        :param arg: numerical argument
        """
        self.stack_pointer += 1
        self.stack[self.stack_pointer] = arg
        self.program_counter += 1

    def ret(self):
        """ Pop address from stack and set PC to this address """
        address = self.pop()
        self.program_counter = address

    def stop(self):
        """ Exit program """
        self.running = False
