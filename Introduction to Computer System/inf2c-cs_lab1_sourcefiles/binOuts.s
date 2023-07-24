 .data
prompt1:        .asciiz  "Enter decimal number.\n"
outmsg:         .asciiz  "The number in binary representation is:\n"
minus:          .asciiz  "-"
        .globl main

        .text
main:   
        # prompt for input
        li   $v0, 4
        la   $a0, prompt1
        syscall

        # Get number from user
        li   $v0, 5
        syscall
        
        add  $s0, $zero, $v0
        
        li   $v0, 4
        la   $a0, outmsg
        syscall
        
        li   $t0, 32
        li $s1,0
        
loop: srl $t1, $s0, 31
      addi  $t1, $t1, 48
      beq $t1,48, jump
      
set$s1: li $s1,1
        j print
      
print: li $v0, 11
       add $a0, $t1, $zero
       syscall
       
jumploop: sll  $s0, $s0, 1   # Drop current leftmost digit
          addi $t0, $t0, -1  # Update loop counter
          bne  $t0, $0, loop # Still not 0?, go to loop
           # end the program
          li   $v0, 10
          syscall
 
jump: beq $s1, $zero, jumploop                            
      j set$s1  
        