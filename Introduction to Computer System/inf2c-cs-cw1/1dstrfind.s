
#=========================================================================
# 1D String Finder 
#=========================================================================
# Finds the [first] matching word from dictionary in the grid
# 
# Inf2C Computer Systems
# 
# Siavash Katebzadeh
# 8 Oct 2019
# 
#
#=========================================================================
# DATA SEGMENT
#=========================================================================
.data
#-------------------------------------------------------------------------
# Constant strings
#-------------------------------------------------------------------------

grid_file_name:         .asciiz  "1dgrid.txt"
dictionary_file_name:   .asciiz  "dictionary.txt"
newline:                .asciiz  "\n"
        
#-------------------------------------------------------------------------
# Global variables in memory
#-------------------------------------------------------------------------
# 
grid:                   .space 33       # Maximun size of 1D grid_file + NULL
.align 4                                # The next field will be aligned
dictionary:             .space 11001    # Maximum number of words in dictionary *
                                        # ( maximum size of each word + \n) + NULL
# You can add your data here!
space:                  .asciiz  " "
negative:               .asciiz  "-1"
.align 4
dicidx:                 .space 11001

#=========================================================================
# TEXT SEGMENT  
#=========================================================================
.text

#-------------------------------------------------------------------------
# MAIN code block
#-------------------------------------------------------------------------

.globl main                     # Declare main label to be globally visible.
                                # Needed for correct operation with MARS
main:
#-------------------------------------------------------------------------
# Reading file block. DO NOT MODIFY THIS BLOCK
#-------------------------------------------------------------------------

# opening file for reading

        li   $v0, 13                    # system call for open file
        la   $a0, grid_file_name        # grid file name
        li   $a1, 0                     # flag for reading
        li   $a2, 0                     # mode is ignored
        syscall                         # open a file
        
        move $s0, $v0                   # save the file descriptor 

        # reading from file just opened

        move $t0, $0                    # idx = 0

READ_LOOP:                              # do {
        li   $v0, 14                    # system call for reading from file
        move $a0, $s0                   # file descriptor
                                        # grid[idx] = c_input
        la   $a1, grid($t0)             # address of buffer from which to read
        li   $a2,  1                    # read 1 char
        syscall                         # c_input = fgetc(grid_file);
        blez $v0, END_LOOP              # if(feof(grid_file)) { break }
        lb   $t1, grid($t0)          
        addi $v0, $0, 10                # newline \n
        beq  $t1, $v0, END_LOOP         # if(c_input == '\n')
        addi $t0, $t0, 1                # idx += 1
        j    READ_LOOP 
END_LOOP:
        sb   $0,  grid($t0)            # grid[idx] = '\0'

        # Close the file 

        li   $v0, 16                    # system call for close file
        move $a0, $s0                   # file descriptor to close
        syscall                         # fclose(grid_file)


        # opening file for reading

        li   $v0, 13                    # system call for open file
        la   $a0, dictionary_file_name  # input file name
        li   $a1, 0                     # flag for reading
        li   $a2, 0                     # mode is ignored
        syscall                         # fopen(dictionary_file, "r")
        
        move $s0, $v0                   # save the file descriptor 

        # reading from  file just opened

        move $t0, $0                    # idx = 0

READ_LOOP2:                             # do {
        li   $v0, 14                    # system call for reading from file
        move $a0, $s0                   # file descriptor
                                        # dictionary[idx] = c_input
        la   $a1, dictionary($t0)       # address of buffer from which to read
        li   $a2,  1                    # read 1 char
        syscall                         # c_input = fgetc(dictionary_file);
        blez $v0, END_LOOP2             # if(feof(dictionary_file)) { break }
        lb   $t1, dictionary($t0)                             
        beq  $t1, $0,  END_LOOP2        # if(c_input == '\0')
        addi $t0, $t0, 1                # idx += 1
        j    READ_LOOP2
        
        
END_LOOP2:
        sb   $0,  dictionary($t0)       # dictionary[idx] = '\0'

        # Close the file 

        li   $v0, 16                    # system call for close file
        move $a0, $s0                   # file descriptor to close
        syscall                         # fclose(dictionary_file)
#------------------------------------------------------------------
# End of reading file block.
#------------------------------------------------------------------

# You can add your code here!
calcnum: li $t9, 0 #the number of words in dictionary
         li $t0, 0
         li $s7, 0#boolean to decide whether to print -1
         
loop:    lb $a0, dictionary($t0)
         beq $a0, 10, incre
         beq $a0 $0, calcdicidx
         addi $t0, $t0, 1
         j loop         
         
incre:  addi $t9, $t9, 1
        addi $t0, $t0, 1
        j loop
 
calcdicidx: li $t0, 0
            li $t1, 0
            sb $t1, dicidx($t0)

loop2:      lb $a0, dictionary($t1)
            beq $a0, 10, addone
            beq $a0, $0, strfind
            addi $t1, $t1, 1
            j loop2
            
addone:     addi $t0, $t0, 1
            addi $t1, $t1, 1
            sb $t1, dicidx($t0)      
            j loop2      
 
 
#----------------------Main part starts here!
 
strfind: li $t0, 0 #idx=0
         li $t1, 0 #grid_idx=0
         la $t2, dictionary
         la $t3, grid
         la $t5, grid

         
while_loop:  lb $a1, grid($t1)
             bne $a1, $0, for_loop#while (grid[grid_idx] != '\0')
             j printneg           #print_int(-1)


for_loop:    lb $a0, dicidx($t0)#dictionary_idx[idx]
             add $t2, $t2, $a0#word=dictionary+dictionary_idx[idx]
             move $t4, $t2    #make a copy of $t2
             add $t3, $t3, $t1#grid+grid_idx
             move $s3, $t3    #make a copy of $t3
                            
contain:     lb $s1, 0($s3)
             lb $s2, 0($t2)
             bne $s1, $s2, check10 #if (*string != *word)
             addi $s3, $s3, 1     #string++
             addi $t2, $t2, 1     #word++
             j contain                     
 
check10:     beq $s2, 10, print#if(*word=='\n), print the result
             addi $t0, $t0, 1  #idx++
             beq  $t0, $t9, grid_idx#if(idx==dic_num_ofword), break for_loop, start new while_loop
             la $t2, dictionary
             la $t3, grid
             j for_loop

grid_idx:    addi $t1, $t1, 1
             la $t3, grid
             li $t0, 0
             j while_loop
                                                                                                                                                                                               
print:   li $v0, 1
         move $a0, $t1
         syscall
         li $v0, 11
         li $a0, 32
         syscall
         addi $s7, $s7, 1

print1:  li $v0, 11
         lb $a0, 0($t4)       
         beq  $a0, 10, for_loop1#continue checking for valid words
         syscall
         addi $t4, $t4, 1      
         j print1
 
for_loop1: li $v0, 11
           li $a0, 10
           syscall
           addi $t0, $t0, 1  #idx++
           la $t2, dictionary#restore the value of $t2
           la $t3, grid      #restore the value of $t3
           j for_loop
                 
printneg: beqz $s1, printneg2
          j main_end

printneg2:  li $v0, 4
            la $a0, negative
            syscall               


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#------------------------------------------------------------------
# Exit, DO NOT MODIFY THIS BLOCK
#------------------------------------------------------------------
main_end:      
        li   $v0, 10          # exit()
        syscall
     
#----------------------------------------------------------------
# END OF CODE
#----------------------------------------------------------------
