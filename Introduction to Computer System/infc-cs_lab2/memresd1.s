# memread0.s program for MIPS


    .data

inp: .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
out: .space 64
msg:.asciiz     "Reading array\n"
newline:.asciiz "\n"

    .globl main  # declare the global symbols of this program
                 # SPIM requires a "main" symbol, (think of symbol
                 #  as another name for label), which declares
                 #  where our program starts

    .text
                 # .text starts the text segment of the program,
                 # where the assembly program code is placed.

main:   # This is the entry point of our program
	  
    la    $a0, msg # make $a0 point to where the message is
    li    $v0, 4   # $v0 <- 4
    syscall        # Call the OS to print the message

    la $a1, inp   # array to be processed
    li $a2, 16     # size of array to be processed          # method to process an array
   
initarr:
	li  $s1,0  #s1 will be the index of the our put array
     	li  $s2,16 # bound of index
     	li  $s3,0
     	la  $s6,out
     	
set:
     	sw $s3,0($s6)
     	addi $s1,$s1,1 #change the index
    	addi $s6,$s6,4
    	jal compare     	
     	
compare:
	slt $s0,$s1,$s2
	bne $s0,$0,set



copyarr:  
        li $s5,0 #$s5 use for index
        addi $s6, $s6, -64
  
copy:
        lw $s3,0($a1)
	sw $s3,0($s6)
     	addi $s5,$s5,1 #change the index
    	addi $s6,$s6,4
    	addi $a1,$a1,4  
    	 
compareCopy:
	slt $s0,$s5,$s2
	bne $s0,$0,copy
        addi $s6,$s6,-64  	
	
loop: beqz  $a2, ending        # go to end if all array elements processed
      lw    $a0, 0($s6)     # load array element into reg $a0
      li    $v0, 1
      syscall               # print element
      la    $a0, newline
      li    $v0, 4
      syscall
      addi  $a2, $a2, -1    # decrement counter for elements left to be processed
      addi  $s6, $s6, 4     # increment address for next element
      j     loop            # end of iteration
       
ending:  

    
    # This is the standard way to end a program
	li    $v0, 10
      	syscall        # end the program
