# memread0.s program for MIPS


    .data

inp: .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
out: .space 64
msg:.asciiz     "Reading array\n"
newline:.asciiz "\n"

    .globl main  
                 

    .text
  
main:       #��Ĵ����һ���������û�а�main������ǰ�棬���������е�ʱ���������loop����loop��Ӧ�������һ�����еģ����Ծ�ʲôҲ�ò���
	  
    la    $a0, msg 
    li    $v0, 4   
    syscall        

    la $a1, inp    # ���������Ĵ��룬$a1ָ����inp�ĵ�ַ������out
    li $a2, 16     
    jal initarr    #�Ҹ�������뷨����init��copy,���print���Ҿ�������ֱ������initarr�������Ͳ����޸�loop��λ�á�
    
    #jal readarr              #���������readarrȥ���ˣ���Ϊ��������û��ʲôʵ������                  

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
               
    #����Ҳ����Ҫʲôend�ˣ�ֱ��ending����������

   
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
    	
        #initarr�ⲿ��д��û������

copyarr:  
        li $s5,0 
        addi $s6, $s6, -64    #��������ֻ������counter��û������out����ʼ��ַ����Ϊout��64��byte������Ҫ����64.��������֮��ſ��Խ���copyarr
        
compareCopy:
	slt $s0,$s5,$s2
	bne $s0,$0,copy
	addi $s6, $s6, -64   #ͬ�������⣬����print��ʱ��ͬ��ҲҪ����out����ʼ��ַ
	jal loop             #��ȥʲôending�ˣ���print�����͸Ͽ�ȥloop
copy:
        lw $s3,0($a1)
	sw $s3,0($s6)
     	addi $s5,$s5,1 
    	addi $s6,$s6,4
    	addi $a1,$a1,4
    	jal compareCopy
    
        #copyarr�ⲿ��Ҳû��ʲô������
ending:     
	li    $v0, 10
      	syscall        
