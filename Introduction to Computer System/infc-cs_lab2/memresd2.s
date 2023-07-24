# memread0.s program for MIPS


    .data

inp: .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
out: .space 64
msg:.asciiz     "Reading array\n"
newline:.asciiz "\n"

    .globl main  
                 

    .text
  
main:       #你的代码第一个问题就是没有把main放在最前面，所以在运行的时候会先运行loop，而loop本应该是最后一个运行的，所以就什么也得不到
	  
    la    $a0, msg 
    li    $v0, 4   
    syscall        

    la $a1, inp    # 根据你后面的代码，$a1指的是inp的地址，不是out
    li $a2, 16     
    jal initarr    #我根据你的想法，先init再copy,最后print，我就在这里直接跳到initarr，这样就不用修改loop的位置。
    
    #jal readarr              #我在这里把readarr去掉了，因为它本来就没有什么实际意义                  

loop:
    beqz  $a2, ending        
    lw    $a0, 0($s6)     
    li    $v0, 1
    syscall               
    la    $a0, newline
    li    $v0, 4
    syscall
    addi  $a2, $a2, -1    
    addi  $s6, $s6, 4     
    j     loop            
               
    #这里也不需要什么end了，直接ending，结束完事

   
initarr:
	li  $s1,0  
     	li  $s2,16 
     	addi $s3,$s3,0     
     	la $s6,out
     	
compare:
	slt $s0,$s1,$s2
	bne $s0,0,set
	jal copyarr
set:
     	sw $s3,0($s6)
     	addi $s1,$s1,1 
    	addi $s6,$s6,4
    	jal compare
    	
        #initarr这部分写的没有问题

copyarr:  
        li $s5,0 
        addi $s6, $s6, -64    #在这里你只重置了counter，没有重置out的起始地址，因为out是64个byte，所以要减掉64.重置完了之后才可以进行copyarr
        
compareCopy:
	slt $s0,$s5,$s2
	bne $s0,$0,copy
	addi $s6, $s6, -64   #同样的问题，你在print的时候同样也要重置out的起始地址
	jal loop             #别去什么ending了，想print出来就赶快去loop
copy:
        lw $s3,0($a1)
	sw $s3,0($s6)
     	addi $s5,$s5,1 
    	addi $s6,$s6,4
    	addi $a1,$a1,4
    	jal compareCopy
    
        #copyarr这部分也没有什么大问题
ending:     
	li    $v0, 10
      	syscall        
