# Synopsis
Create a Computer Simulator, where given the following program:

```c
function add_disp(){
  print(add(1,2));
}
add_disp();
print(777);
```

the compiler translates it into machine code, while executing the aforementioned functions.

# Usage

Two strategies are used for solving this problem. Initially, a single stack memory is used where both instructions and values are pushed and popped to the same stack. Then, a slightly different approach was considered, with independent stack and instructions memories. 

The following approaches are represented, respectively, by the .py sets described below:

Main.py / Computer.py

```python
import Computer

def main()
   
  code = Computer.Computer(np.array([None]*100))
  code.insertinstruction(instruction_name, instruction_arg)
  ...
  code.execute()
  
main()
```

MainDoubleMemory.py / ComputerDoubleMemory.py

```python
import ComputerDoubleMemory

def main()
   
  code = ComputerDoubleMemory.ComputerDoubleMemory(np.array([None]*100), np.array([None]*100))
  code.insertinstruction(instruction_name, instruction_arg)
  ...
  code.execute()
  
main()
```
