# Hello world program for MIPS/SPIM
# 
# Written by Aris Efthymiou, 16/08/2005

# As you have guessed, this is a comment, up to the end of the line.

    .data
                 #  .data starts the data segment of the program,
                 # where all the global variables are held.
msg:.asciiz    "Hello world!\n"

                 # "msg" is a label, a name followed by colon, ":".
                 #   It must start at the beginning of the line.
                 #   In contrast, all assembly directives (e.g. ".data")
                 # must *not* start at the beginning of the line.
                 #  .asciiz stores a string (ending with NULL) in memory

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

    # This is the standard way to end a program
    li    $v0, 10
    syscall        # end the program
