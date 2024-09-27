import json 
import time
import sys
from copy import deepcopy

def to_uint64(n):
    if isinstance(n, list):
        return [x & 0xFFFFFFFFFFFFFFFF for x in n]
    return n & 0xFFFFFFFFFFFFFFFF


def parse_instruction(instruction):
    parts = instruction.replace(',', '').split()
    opcode = parts[0] # Extract the opcode
    is_imm = False # Initialize the is_imm flag as False
    rd = int(parts[1][1:])# Extract the destination register number (removing 'x' prefix)
    rs1 = int(parts[2][1:]) # Extract the source register 1 number (removing 'x' prefix)
    rs2_or_imm = parts[3]# Initialize rs2 or the immediate value
    if rs2_or_imm.startswith('x'):# Check if the last part is a register or an immediate value
        # If it's a register, extract the register number
        rs2 = int(rs2_or_imm[1:])
        is_imm = False
    else:
        # If it's an immediate value, convert to integer
        rs2 = int(rs2_or_imm)
        is_imm = True
    return opcode, rd, rs1, rs2, is_imm

def is_imm(opcode):
    return opcode in ['addi', 'subi', 'muli', 'divi', 'remi']

def empty_json_log(output_file):
    with open(output_file, 'w') as file:
        file.write('')

class Instruction:
    def __init__(self, json_string, pc, **kwargs):
        # Instruction parsing
        self.opcode, self.rd, self.rs1, self.rs2, self.is_imm = self.parse_instruction(json_string)
        self.pc = pc
        
        # Active List attributes
        self.active_done = kwargs.get('active_done', False)
        self.active_exception = kwargs.get('active_exception', False)
        self.active_log_dest = kwargs.get('active_log_dest', None)
        self.active_old_dest = kwargs.get('active_old_dest', None)
        
        # Queue attributes
        self.queue_dest = kwargs.get('queue_dest', None)
        self.queue_op_a_is_ready = kwargs.get('queue_op_a_is_ready', False)
        self.queue_op_a_reg = kwargs.get('queue_op_a_reg', None)
        self.queue_op_b_is_ready = kwargs.get('queue_op_b_is_ready', False)
        self.queue_op_b_reg = kwargs.get('queue_op_b_reg', None)
        self.queue_op_a_value = kwargs.get('queue_op_a_value', 0)
        self.queue_op_b_value = kwargs.get('queue_op_b_value', 0)
        self.queue_op_a_tag = kwargs.get('queue_op_a_tag', 0)
        self.queue_op_b_tag = kwargs.get('queue_op_b_tag', 0)
        self.queue_a_forward = kwargs.get('queue_a_forward', False)
        self.queue_b_forward = kwargs.get('queue_b_forward', False)
        self.queue_a_forward_value = kwargs.get('queue_a_forward_value', 0)
        self.queue_b_forward_value = kwargs.get('queue_b_forward_value', 0)
        
        # ALU attributes
        self.alu_result = kwargs.get('alu_result', None)
        self.alu_exception = kwargs.get('alu_exception', False)
        self.alu_number = kwargs.get('alu_number', None)
    
    @staticmethod
    def parse_instruction(instruction):
        parts = instruction.replace(',', '').split()
        opcode = parts[0]
        is_imm = False
        rd = int(parts[1][1:])
        rs1 = int(parts[2][1:])
        rs2_or_imm = parts[3]
        if rs2_or_imm.startswith('x'):
            rs2 = int(rs2_or_imm[1:])
        else:
            rs2 = int(rs2_or_imm)
            is_imm = True
        return opcode, rd, rs1, rs2, is_imm

    def active_to_dict(self):
        """Convert active list relevant attributes to a dictionary."""
        return {
            'Done': self.active_done,
            'Exception': self.active_exception,
            'LogicalDestination': self.active_log_dest,
            'OldDestination': self.active_old_dest,
            'PC': self.pc
        }
        
    def queue_to_dict(self):
        """Convert queue relevant attributes to a dictionary."""
        opcode = deepcopy(self.opcode)
        if opcode == 'addi':
            opcode = 'add'
        return {
            'DestRegister': self.queue_dest,
            'OpAIsReady': self.queue_op_a_is_ready,
            'OpARegTag': self.queue_op_a_tag,
            'OpAValue': to_uint64(self.queue_op_a_value),
            'OpBIsReady': self.queue_op_b_is_ready,
            'OpBRegTag': self.queue_op_b_tag,
            'OpBValue': to_uint64(self.queue_op_b_value),
            'OpCode': opcode,
            'PC': self.pc
        }

class OoO470:
    def __init__(self, copy=False):
        self.pc =0
        self.physical_registers = [0]*64
        self.decoded_instruction_register = []  # register 1 figure
        self.exception_flag = False
        self.exception_pc = 0
        self.exception_pc_reg = 0
        self.exception_handler_done = False
        self.register_map_table = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        self.free_list = [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
        self.busy_bit_table = [False]*64
        self.active_list = []
        self.active_list_to_clear = []
        self.registers_to_clear = []  
        self.integer_queue = []   # pipeline stage 2
        self.copy_integer_queue = []
        self.remaining_instructions = []
        self.pipeline_stage1 = []
        self.pipeline_stage_3 = []
        self.pipeline_stage_4 = []
        self.pipeline_stage5 = []
        self.copy_pipeline_stage5 = []
        self.exception_commit_reg=False
        self.backpressure = False
        self.copy = copy
        self.state = []
        self.new_active_list = []
        self.done_instructions = []

    def initialize(self):
        self.pc = 0
        self.physical_registers = [0]*64
        self.decoded_instruction_register = []
        self.exception_flag = False
        self.exception_pc = 0
        self.exception_pc_reg = 0
        self.exception_handler_done = False
        self.register_map_table = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        self.free_list = [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
        self.busy_bit_table = [False]*64
        self.active_list = []
        self.active_list_to_clear = []
        self.registers_to_clear = []
        self.integer_queue = []
        self.copy_integer_queue = []
        self.remaining_instructions = []
        self.pipeline_stage1 = []
        self.pipeline_stage_3 = []
        self.pipeline_stage_4 = []
        self.pipeline_stage5 = []
        self.copy_pipeline_stage5 = []
        self.exception_commit_reg = False
        self.backpressure = False
        self.copy = False
        self.state = []
        self.new_active_list = []
        self.done_instructions = []

    def dumpStateIntoLog(self):
        cycle_state = {
            'PC': deepcopy(self.pc),
            'PhysicalRegisterFile': deepcopy(self.physical_registers),
            'DecodedPCs': deepcopy(self.decoded_instruction_register),
            'ExceptionPC' : deepcopy(self.exception_pc),
            'Exception' : deepcopy(self.exception_flag),
            'RegisterMapTable': to_uint64(deepcopy(self.register_map_table)),
            'FreeList': deepcopy(self.free_list),
            'BusyBitTable': deepcopy(self.busy_bit_table),
            'ActiveList': [entry.active_to_dict() for entry in deepcopy(self.active_list)],
            'IntegerQueue': [entry.queue_to_dict() for entry in deepcopy(self.integer_queue)],
        }
        self.state.append(deepcopy(cycle_state))

    def writeLogToFile(self, file_path):
        with open(file_path, 'w') as log_file:  # 'a' is for appending to the file
            json.dump(self.state, log_file, indent=2)

    def load_program(self, program_file):
            with open(program_file, 'r') as file:
                instructions = json.load(file)
                self.remaining_instructions = [instruction for instruction in instructions] # Ensure that instructions are parsed as dictionaries

    def remove_by_pc(self, pc, lst):
        for entry in lst:
            if entry.pc == pc:
                lst.remove(entry)
                return

    def fetch_and_decode(self):
        rem=len(self.remaining_instructions)
        if(self.exception_flag==True): # if there is an exception, disable the fetch and decode stage (forwarding from commit)
            self.pc=0x10000  # jump to the exception handler
            self.pipeline_stage1.clear()    # clear the decoded instruction register
            self.decoded_instruction_register.clear()
        else:
            i=1 
            while i<=4 and i<=rem and not self.backpressure: # fetch up to 4 instructions
                instruction_str = self.remaining_instructions.pop(0)
                self.pipeline_stage1.append(Instruction(instruction_str,deepcopy(self.pc))) 
                self.decoded_instruction_register.append(self.pc)
                self.pc = self.pc + 1 # increment the program counter
                i=i+1

    def rename_and_dispatch(self,copy=False):
        self.backpressure = False
        #  Observe the results of all functional units through the forwarding paths and update the physical 
        # register file as well as the Busy Bit Table.
        for alu_out in self.pipeline_stage5:
            if (not alu_out.alu_exception) and (not self.exception_flag) and (not alu_out.alu_result is None):
                self.physical_registers[alu_out.queue_dest] = to_uint64(alu_out.alu_result) # update physical register file
                self.busy_bit_table[alu_out.queue_dest] = False              # update busy bit table

        active_list_length = len(self.active_list)
        free_list_length = len(self.free_list)
        integer_queue_length = len(self.integer_queue)
        length = len(self.pipeline_stage1)
        if (active_list_length <= 28 and free_list_length >= 4 and integer_queue_length<=28 and (not self.exception_flag) ): # if there is space in the active list and the free list and the integer queue (instruction treated atomically)
            for i in range(length): #take instructions from pipeline stage 1
                instruction = self.pipeline_stage1.pop(0)  
                self.decoded_instruction_register.pop(0)
                log_dest = instruction.rd
                old_dest = self.register_map_table[log_dest]
                #if(copy==True):
                instruction.active_old_dest = old_dest
                instruction.active_log_dest = log_dest
                self.active_list.append(instruction) # add the decoded instruction to the active list
                
                # apply renaming and set ready flags and busy bits
                op_a = self.register_map_table[instruction.rs1]
                if (not self.busy_bit_table[op_a]):
                    ready_a = True
                    tag_a = 0
                    value_a=self.physical_registers[op_a]
                else:
                    ready_a = False
                    tag_a = op_a
                    value_a = 0

                if not instruction.is_imm: 
                    op_b = self.register_map_table[instruction.rs2]
                    if (not self.busy_bit_table[op_b]):
                        ready_b = True
                        tag_b = 0
                        value_b = self.physical_registers[op_b]
                    else:
                        ready_b = False
                        tag_b = op_b
                        value_b = 0
                else:
                    op_b = instruction.rs2
                    ready_b = True
                    tag_b = 0
                    value_b = op_b

                physical_dest = self.free_list[0]     #   get a free physical register
                self.free_list.pop(0)                 #   remove the physical register from the free list
                self.register_map_table[log_dest] = physical_dest # update register map table
                self.busy_bit_table[physical_dest] = True # set the busy bit of the physical register
                instruction.queue_dest = physical_dest
                instruction.queue_op_a_is_ready = ready_a
                instruction.queue_op_a_reg = op_a
                instruction.queue_op_b_is_ready = ready_b
                instruction.queue_op_b_reg = op_b
                instruction.queue_op_a_tag = tag_a
                instruction.queue_op_b_tag = tag_b
                instruction.queue_op_a_value = value_a
                instruction.queue_op_b_value = value_b
                self.integer_queue.append(instruction) # add decoded and renamed instruction to the integer queue
                if(i==3):    # decode up to 4 instructions
                    break
        else:
            self.backpressure = True # if there is no space in the active list or the free list or the integer queue, set the backpressure flag

    def issue(self):
        ready_instructions = []
        issued_instructions = []
        # forwarding paths (check ready operands from execute output)
        for alu_out in self.pipeline_stage5: # iterate over completed instructions
            if alu_out.alu_exception==False:  # if there is no exception in the forwarded instruction
                for entry in self.integer_queue:
                    if entry.queue_op_a_reg == alu_out.queue_dest:
                        entry.queue_a_forward = True
                        entry.queue_a_forward_value = alu_out.alu_result
                        entry.queue_op_a_is_ready = True
                        entry.queue_op_a_tag = 0
                        entry.queue_op_a_value = alu_out.alu_result
                    if ((entry.queue_op_b_reg == alu_out.queue_dest) and ( is_imm(entry.opcode) == False)):
                        entry.queue_b_forward = True
                        entry.queue_b_forward_value = alu_out.alu_result
                        entry.queue_op_b_is_ready = True
                        entry.queue_op_b_tag = 0
                        entry.queue_op_b_value = alu_out.alu_result

        # send 4 oldest ready instructions to the execution units
        for entry in self.integer_queue:
            if (entry.queue_op_a_is_ready or entry.queue_a_forward) and (entry.queue_op_b_is_ready or entry.queue_b_forward):  # ready check
                ready_instructions.append(entry)
        i=1
        n_ready = len(ready_instructions)
        while i<= n_ready and i<=4: # issue up to 4 instructions
            new_entry=ready_instructions[0]
            for entry in ready_instructions:
                if entry.pc < new_entry.pc:
                    new_entry = entry
            ready_instructions.remove(new_entry)
            issued_instructions.append(new_entry)
            i=i+1
        for i , entry in enumerate(issued_instructions):
            if (entry.queue_a_forward):
                entry.queue_a_value=entry.queue_a_forward_value
            else:
                entry.queue_a_value=entry.queue_op_a_value
            if (entry.queue_b_forward):
                entry.queue_b_value=entry.queue_b_forward_value
            else:
                entry.queue_b_value=entry.queue_op_b_value
            entry.alu_number = i
            self.pipeline_stage_3.append(entry)
            self.remove_by_pc(entry.pc, self.integer_queue)
                        
    def execute(self, copy=False):
        for entry in self.pipeline_stage_4:
            if entry.opcode == "add":
                entry.alu_result = to_uint64(entry.queue_a_value) + to_uint64(entry.queue_b_value)
            elif entry.opcode == "addi":
                entry.alu_result = to_uint64(entry.queue_a_value) + to_uint64(entry.queue_b_value)
            elif entry.opcode == "sub":
                entry.alu_result = to_uint64(entry.queue_a_value) - to_uint64(entry.queue_b_value)
            elif entry.opcode == "mulu":
                entry.alu_result = to_uint64(entry.queue_a_value) * to_uint64(entry.queue_b_value)
            elif entry.opcode == "divu":
                if to_uint64(entry.queue_b_value) == 0:
                    entry.alu_exception = True
                else:
                    entry.alu_result = to_uint64(entry.queue_a_value) // to_uint64(entry.queue_b_value) # ??? / or //
            elif entry.opcode == "remu":
                if to_uint64(entry.queue_b_value) == 0:
                    entry.alu_exception = True
                else:
                    entry.alu_result = to_uint64(entry.queue_a_value) % to_uint64(entry.queue_b_value)
        if (copy==True): # modify the pipeline stage 5 only once in the propagate function (it gets computed by the copy of the processor)
            self.pipeline_stage5.extend(self.pipeline_stage_4)
        self.pipeline_stage_4 = []
        self.pipeline_stage_4.extend(self.pipeline_stage_3)
        self.pipeline_stage_3 = []

    def commit(self):
        #forwarding from execution stage to commit stage
        if (self.exception_commit_reg==False): # if there is no exception, commit the instructions
            self.exception_flag = False
            for committed_inst in self.pipeline_stage5: #  update done flags in the active list from forwarding
                if committed_inst.alu_exception == True: # exception check from frowarding
                        self.exception_commit_reg = True
                        # self.copy_integer_queue = deepcopy(self.integer_queue)
                        # self.copy_pipeline_stage5 = deepcopy(self.pipeline_stage5)
                        self.exception_pc_reg = committed_inst.pc # done the next cycle in exception mode
                        for active_inst in self.active_list:
                            if (active_inst.pc == committed_inst.pc):
                                active_inst.active_exception = True
                                active_inst.active_done = True
                                
                for active_inst in self.active_list:
                    if (active_inst.pc == committed_inst.pc):  # done check from forwarding
                        active_inst.active_done = True    
            picked_inst = 1
            n_active = len(self.active_list)
            copy_active_list = deepcopy(self.active_list) 
            while picked_inst <= 4 and picked_inst <= n_active:
                oldest = copy_active_list[0]
                for active_inst in copy_active_list:
                    if active_inst.pc < oldest.pc:
                        oldest  = active_inst
                if ((oldest.active_done == True) ):
                    self.busy_bit_table[oldest.active_old_dest] = False  # done in rename and dispatch stage
                    picked_inst = picked_inst + 1
                    if (oldest.active_exception == False):
                        self.active_list_to_clear.append(oldest)
                        copy_active_list.remove(oldest)
                else:
                    break

        # TODO : keep modifying the code so that it only uses the class instruction. major problem: deepcopies

        else: # exception handler
            self.exception_flag = True
            self.exception_pc = self.exception_pc_reg # record the pc of the instruction that caused the exception
            self.pipeline_stage1.clear()
            self.pipeline_stage_3.clear() # the execution units are cleared
            self.pipeline_stage_4.clear() # the execution units are cleared
            self.pipeline_stage5.clear() # the execution units are cleared
            self.integer_queue.clear() # reset the integer queue
            i=1
            length = len(self.active_list)
            copy_active_list = deepcopy(self.active_list)
            while i<=4 and i<=length:
                newest = copy_active_list[0]
                for active_inst in copy_active_list: # look for the newest instruction in the active list
                    if active_inst.pc > newest.pc:
                        newest = active_inst
                
                self.registers_to_clear.append([newest.queue_dest,newest.active_old_dest,newest.active_log_dest,newest.pc]) # [new_dest , old_dest , log_dest, pc]
                self.active_list_to_clear.append(newest) # remove the newest instruction from the active list
                copy_active_list.remove(newest)
                i=i+1
            if (len(copy_active_list)==0) and (len(self.active_list_to_clear)==0) and (len(self.registers_to_clear)==0):  # if there are no more instructions in the active list exit the exception handler
                self.exception_commit_reg = False
                self.exception_handler_done = True

    def clear_active_list(self):
        if (self.exception_flag==False): # if there is no exception, clear the active list
            for entry in list(self.active_list_to_clear):  # Iterate over a copy of the list
                self.remove_by_pc(entry.pc, self.active_list)
                self.free_list.append(entry.active_old_dest)  # free the physical register

        if (self.exception_commit_reg==True): # if there is an exception, clear the active list and the registers to clear (forwarding from commit stage
            for entry in list(self.active_list_to_clear):  # Iterate over a copy of the list
                self.remove_by_pc(entry.pc, self.active_list)
            copy_registers_to_clear = deepcopy(self.registers_to_clear) 
            n=1    
            length = len(copy_registers_to_clear)
            while n<=4 and n<=length: # clear up to 4 registers
                newest = copy_registers_to_clear[0]  
                for reg in copy_registers_to_clear:
                    if reg[3] > newest[3]:
                        newest = reg
                self.busy_bit_table[newest[0]] = False
                self.free_list.append(newest[0])
                self.register_map_table[newest[2]] = newest[1]
                copy_registers_to_clear.remove(newest)
                n=n+1
        self.active_list_to_clear.clear() 
        self.registers_to_clear.clear()

    def propagate(self):
        copy = deepcopy(self)
        copy.execute(copy=True)
        self.pipeline_stage5=deepcopy(copy.pipeline_stage5)
        self.clear_active_list()
        self.commit()
        self.execute(copy=False) 
        self.issue()
        self.rename_and_dispatch()
        self.fetch_and_decode()
        self.pipeline_stage5.clear()
        return self
    

if __name__ == "__main__":
    if len(sys.argv) != 3:  # Corrected to check for exactly 3 arguments: The script name, input file, and output file
        print("Usage: python simulator.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    processor = OoO470()
    processor.initialize()
    processor.load_program(input_file)  # Load the program from the specified input file
    processor.dumpStateIntoLog() # cycle 0
    ex=0
    cycles=0
    while (not (len(processor.remaining_instructions) == 0 and len(processor.active_list) == 0)) or (ex<2 and processor.exception_handler_done):
        processor.propagate()
        processor.dumpStateIntoLog()
        if(processor.exception_handler_done==True):
            ex=ex+1
    processor.writeLogToFile(output_file) 
    print("Simulation completed successfully!") # Print a success message

