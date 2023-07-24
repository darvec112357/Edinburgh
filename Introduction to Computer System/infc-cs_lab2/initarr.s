    .data

inp: .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
out: .space 64

msg1:.asciiz    "Initialise arr\n"

newline:.asciiz "\n"

    .globl main  # declare the global symbols of this program
                 # SPIM requires a "main" symbol, (think of symbol
                 #  as another name for label), which declares
                 #  where our program starts

    .text
    
main:   # This is the entry point of our program

    la    $a0, msg1 # make $a0 point to where the message is
    li    $v0, 4   # $v0 <- 4
    syscall        # Call the OS to print the message

    la $a1, inp    # array to be processed
    la $a2, out
    li $a3, 16

    jal initarr
    
    # This is the standard way to end a program
    li    $v0, 10
    syscall        # end the program
    
initarr:

loop1: beqz $a3, set$a3
       li $t1, 0
       sw $t1, 0($a2)
       addi  $a3, $a3, -1   
       addi  $a2, $a2, 4     
       j     loop1
  
set$a3: addi $a3, $a3, 16   
        addi $a2, $a2, -64  
            
loop:
    beqz  $a3, end        # go to end if all array elements processed
    lw    $a0, 0($a2)     # load array element into reg $a0
    li    $v0, 1
    syscall               # print element
    la    $a0, newline
    li    $v0, 4
    syscall
    addi  $a3, $a3, -1    # decrement counter for elements left to be processed
    addi  $a2, $a2, 4     # increment address for next element
    j     loop            # end of iteration
end:   
    jr $ra                # return





