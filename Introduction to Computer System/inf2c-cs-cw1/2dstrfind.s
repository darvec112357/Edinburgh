
#=========================================================================
# 2D String Finder 
#=========================================================================
# Finds the matching words from dictionary in the 2D grid
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

grid_file_name:         .asciiz  "2dgrid.txt"
dictionary_file_name:   .asciiz  "dictionary.txt"
newline:                .asciiz  "\n"
        
#-------------------------------------------------------------------------
# Global variables in memory
#-------------------------------------------------------------------------
# 
grid:                   .space 1057     # Maximun size of 2D grid_file + NULL (((32 + 1) * 32) + 1)
.align 4                                # The next field will be aligned
dictionary:             .space 11001    # Maximum number of words in dictionary *
                                        # ( maximum size of each word + \n) + NULL
# You can add your data here!
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
         li $s5, 0#row_num
         li $s6, 0#col_num
         
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
            beq $a0, $0, loop3
            addi $t1, $t1, 1
            j loop2
            
addone:     addi $t0, $t0, 1
            addi $t1, $t1, 1
            sb $t1, dicidx($t0)      
            j loop2      
 
loop3:      li $t0, 0
            
loop4:      lb   $a0, dictionary($t0)#loop for calculating number of rows
            beq  $a0, 10, addrow
            beq  $a0, $0, reseti
            addi $t0, $t0, 1
            j loop4            
            
addrow:     addi $s5, $s5, 1
            addi $t0, $t0, 1      
            j loop4

reseti:     li $t0, 0
                        
loop5:      lb $a0, grid($t0)#loop for calculating number of cols
            beq $a0, 10, strfind
            addi $t0, $t0, 1
            addi $s6, $s6, 1 
            j loop5                     
#----------------------Main part starts here!
 
strfind: li $t0, 0 #idx=0
         li $t1, 0 #grid_idx=0
         la $t2, dictionary
         la $t3, grid
         li $t7, 0#int row=0
         li $t8, 0#int col=0
         li $s0, 0#counter for containWH, containWV
         li $v1, 0#indicator to check whether to print(-1)

                  
while_loop:  lb $a1, grid($t1)
             bne $a1, $0, setrc#while (grid[grid_idx] != '\0')
             j printneg           #print_int(-1)

setrc:       beq $a1, 10, addrc
             j for_loop
             
addrc:       addi $t7, $t7, 1
             li $t8,0
             addi $t1, $t1, 1
             j while_loop             

while_loop1: addi $t8, $t8, 1#col++
             addi $t1, $t1, 1#grid_idx++
             li $t0, 0
             j while_loop
             
for_loop:    beq $t0, $t9, while_loop1
             lb $a0, dicidx($t0) #dictionary_idx[idx]
             add $t2, $t2, $a0 #word=dictionary+dictionary_idx[idx]
             addi $t4, $t2, 0    #make a copy of $t2
             add $t3, $t3, $t1 #grid+grid_idx, $t2$t3在这里都是指针
             addi $s3, $t3, 0    #make a copy of $t3
             jal containH 
             addi $t4, $t2, 0    #make a copy of $t2
             addi $s3, $t3, 0    #make a copy of $t3
             bne $a0,$0,printH

             
checkV:      addi $t4, $t2, 0 
             addi $s3, $t3, 0
             jal containV
             addi $t4, $t2, 0    #make a copy of $t2
             addi $s3, $t3, 0    #make a copy of $t3
             bne $a0, $0, printV #if(contain(*string, *word) print

checkD:      addi $t4, $t2, 0 
             addi $s3, $t3, 0
             jal containD
             addi $t4, $t2, 0    #make a copy of $t2
             addi $s3, $t3, 0    #make a copy of $t3
             bne $a0, $0, printD #if(contain(*string, *word) print     
             j refor_loop                                 
                                                                                                                     
containH:    lb $s1, 0($s3)   #*string
             lb $s2, 0($t4)   #*word
             beq $s1 10, containH2  #if (*string =='\n'&& *word =='\n')
             
containH1:   bne $s1, $s2, check10 #if (*string != *word)
             addi $s3, $s3, 1     #string++
             addi $t4, $t4, 1     #word++
             j containH     
                             
containH2:   beq $s2, 10, return1
             j containH1
                        
containV:    lb $s1, 0($s3)   #*string
             lb $s2, 0($t4)   #*word
 	     bne $s1, $s2, check10 #if(*string != *word)         
             addi $t6, $s6, 1
             add  $s3, $s3, $t6 #string+=(row_length+1);
             addi  $t4, $t4, 1   #word++
             j containV
                         
containD:    lb $s1, 0($s3)   #*string
             lb $s2, 0($t4)   #*word
             beq $s1 10, containD2  #if (*string =='\n'&& *word =='\n')
             
containD1:   bne $s1, $s2, check10 #if (*string != *word)
             addi $t6, $s6, 2
             add  $s3, $s3, $t6 #string+=(row_length+2);
             addi $t4, $t4, 1     #word++
             j containD     
                             
containD2:   beq $s2, 10, return1
             j containD1

check10:     beq $s2, 10, return1#return (*word == '\n');
             j return0  
             
return1:     li $a0, 1
             addi $v1, $v1, 1
             jr $ra             

return0:     li $a0, 0
             jr $ra
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
printH:  li $v0, 1
         move $a0, $t7
         syscall
         li $v0, 11
         li $a0, 44
         syscall  
         li $v0, 1
         move $a0, $t8
         syscall 
         li $v0, 11
         li $a0, 32
         syscall  
         li $v0, 11
         li $a0, 72
         syscall  
         li $v0, 11
         li $a0, 32
         syscall
         j printHW                
  
printV:  li $v0, 1
         move $a0, $t7
         syscall
         li $v0, 11
         li $a0, 44
         syscall  
         li $v0, 1
         move $a0, $t8
         syscall 
         li $v0, 11
         li $a0, 32
         syscall  
         li $v0, 11
         li $a0, 86
         syscall  
         li $v0, 11
         li $a0, 32
         syscall
         j printVW

printD:  li $v0, 1
         move $a0, $t7
         syscall
         li $v0, 11
         li $a0, 44
         syscall  
         li $v0, 1
         move $a0, $t8
         syscall 
         li $v0, 11
         li $a0, 32
         syscall  
         li $v0, 11
         li $a0, 68
         syscall  
         li $v0, 11
         li $a0, 32
         syscall
         j printDW                   
                                                                    
printHW: li $v0, 11
         lb $a0, 0($t4)       
         beq  $a0, 10, for_loop1#continue checking for valid words
         syscall
         addi $t4, $t4, 1      
         j printHW
         
printVW: li $v0, 11
         lb $a0, 0($t4)       
         beq  $a0, 10, for_loop2#continue checking for valid words
         syscall
         addi $t4, $t4, 1      
         j printVW      
 
printDW: li $v0, 11
         lb $a0, 0($t4)       
         beq  $a0, 10, println#continue checking for valid words
         syscall
         addi $t4, $t4, 1      
         j printDW  
 
for_loop1: li $v0, 11
           li $a0, 10
           syscall    
           j checkV

println:   li $v0, 11
           li $a0, 10
           syscall
           
refor_loop: addi $t0, $t0, 1#idx++
            la $t2, dictionary
            la $t3, grid
            j for_loop
           
for_loop2: li $v0, 11
           li $a0, 10
           syscall    
           j checkD               
                                                         
printneg:  beqz $v1, printneg2
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
