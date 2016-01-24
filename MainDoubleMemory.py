import numpy as np
import ComputerDoubleMemory

MAIN_BEGIN = 0
ADD_DISPLAY_BEGIN = 50


def main():

    # Instruction memory of maximum 100 characters
    code = ComputerDoubleMemory.ComputerDoubleMemory(np.array([None]*100),
                                                     np.array([None]*100))
    # Insert instructions for ADD_DISPLAY function starting from address
    # ADD_DISPLAY_BEGIN
    code.set_address(ADD_DISPLAY_BEGIN).insert("ADD").insert("PRN").\
        insert("RET")
    # Insert Instructions for MAIN function starting from address MAIN_BEGIN
    code.set_address(MAIN_BEGIN).insert("PUSH", 4)  # This is the return
    # address when execution returns from ADD_DISPLAY
    code.insert("PUSH", 1).insert("PUSH", 2).insert("CALL", ADD_DISPLAY_BEGIN)
    # Next instruction is in address 4:
    code.insert("PUSH", 777).insert("PRN").insert("STOP")
    # Set the Program Counter to the MAIN function and execute
    code.set_address(MAIN_BEGIN).execute()

    return 0

main()
