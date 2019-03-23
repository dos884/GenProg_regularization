class TuringMachine(object):

    def __init__(self,
                 final_states=None,
                 transition_function=None,
                 initial_state="",
                 blank_symbol=" "):

        self.__head_position = 0
        self.__blank_symbol = blank_symbol
        self.__current_state = initial_state
        if transition_function == None:
            self.__transition_function = {}
        else:
            self.__transition_function = transition_function
        if final_states == None:
            self.__final_states = set()
        else:
            self.__final_states = set(final_states)

    def set_tape(self, tape):
        self.__tape = list(tape)

    def get_tape(self):
        return str(self.__tape)

    def step(self):
        char_under_head = self.__tape[self.__head_position]
        x = self.__current_state
        if x in self.__transition_function:
            y = self.__transition_function[x][char_under_head]
            # y is now a tuple: (new state, symbol to write, Left/Right)
            # Write the symbol:
            self.__tape[self.__head_position] = y[1]
            # Move the head
            if y[2] == "R":
                self.__head_position += 1
            elif y[2] == "L":
                self.__head_position -= 1
            # Change the current state:
            self.__current_state = y[0]

    def isFinal(self):
        if self.__current_state in self.__final_states:
            return True
        else:
            return False


initial_state = "init",
accepting_states = ["final"],

transition_function = {"init":
                           {"0": ("init", "1", "R"),
                            "1": ("init", "0", "R"),
                            " ": ("final", " ", "N")}
                       }

final_states = {"final"}

t = TuringMachine(initial_state="init",
                  final_states=final_states,
                  transition_function=transition_function)
t.set_tape("010011 ")

print("Input on Tape:\n" + t.get_tape())

while not t.isFinal():
    t.step()

print("Result of the Turing machine calculation:")
print(t.get_tape())
