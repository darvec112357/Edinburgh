/***********************************************************************
* File       : <wraparound.c>
*
* Author     : <M.R. Siavash Katebzadeh>
*
* Description:
*
* Date       : 08/10/19
*
***********************************************************************/
// ==========================================================================
// 2D String Finder
// ==========================================================================
// Finds the matching words from dictionary in the 2D grid, including wrap-around

// Inf2C-CS Coursework 1. Task 6
// PROVIDED file, to be used as a skeleton.

// Instructor: Boris Grot
// TA: Siavash Katebzadeh
// 08 Oct 2019

#include <stdio.h>

// maximum size of each dimension
#define MAX_DIM_SIZE 32
// maximum number of words in dictionary file
#define MAX_DICTIONARY_WORDS 1000
// maximum size of each word in the dictionary
#define MAX_WORD_SIZE 10

int read_char() { return getchar(); }
int read_int()
{
  int i;
  scanf("%i", &i);
  return i;
}
void read_string(char* s, int size) { fgets(s, size, stdin); }
void print_char(int c)     { putchar(c); }
void print_int(int i)      { printf("%i", i); }
void print_string(char* s) { printf("%s", s); }
void output(char *string)  { print_string(string); }

// dictionary file name
const char dictionary_file_name[] = "dictionary.txt";
// grid file name
const char grid_file_name[] = "2dgrid.txt";
// content of grid file
char grid[(MAX_DIM_SIZE + 1 /* for \n */ ) * MAX_DIM_SIZE + 1 /* for \0 */ ];
// content of dictionary file
char dictionary[MAX_DICTIONARY_WORDS * (MAX_WORD_SIZE + 1 /* for \n */ ) + 1 /* for \0 */ ];
///////////////////////////////////////////////////////////////////////////////
/////////////// Do not modify anything above
///////////////Put your global variables/functions here///////////////////////
int dict_num_words = 0;

int dictionary_idx[MAX_DICTIONARY_WORDS];

int a=0;

int row_length=0;

int gridheight=0;

int gridlength=0;

void print_word(char *word)
{
  while(*word != '\n' && *word != '\0') {
    print_char(*word);
    word++;
  }
}

int containWH(char *string,char *word){
  int i=0;
  while(i<=row_length){
    if(*string=='\n'){
      if(*word=='\n'){
        return 1;
      }
      string=string-row_length;
    }
    if(*string!=*word){
      return (*word == '\n');
    }
    word++;
    string++;
    i++;
  }
  return 0;
}

int containWV(char *string,char *word,int row){
  int i=0;
  int row1=row;
  while(i<=gridheight){
    if(*string!=*word){
      return (*word == '\n');
    }
    if(row1==gridheight-1){
      string=string-((row_length+1)*(gridheight-1));
      word++;
      row1++;
      i++;
      continue;
    }
    word++;
    string+=(row_length+1);
    row1++;
    i++;
  }
  return 0;
}

int containWD(char *string,char *word,int row,int col){
  int row1=row;
  int col1=col;
  while(1){
    if(*string!=*word){
      return (*word == '\n');
    }
    if(row1==gridheight-1||col1==row_length-1){
      while(row1>=0&&col1>=0){
        string-=(row_length+2);
        row1--;
        col1--;
      }
    }
    word++;
    string+=(row_length+2);
    row1++;
    col1++;
  }
  return 0;
}

void strfind()
{
  int idx = 0;
  int grid_idx = 0;
  char *word;
  int row=0;
  int col=0;
  while (grid[grid_idx] != '\0') {
    if(grid[grid_idx] == '\n'){
      row++;
      col=0;
      grid_idx++;
      continue;
    }
    for(idx = 0; idx < dict_num_words; idx ++) {
      word = dictionary + dictionary_idx[idx];
      if (containWH(grid + grid_idx, word)) {
        print_int(row);
        print_char(',');
        print_int(col);
        print_char(' ');
        print_char('H');
        print_char(' ');
        print_word(word);
        print_char('\n');
        a=1;
      }
      if (containWV(grid + grid_idx, word,row)) {
        print_int(row);
        print_char(',');
        print_int(col);
        print_char(' ');
        print_char('V');
        print_char(' ');
        print_word(word);
        print_char('\n');
        a=1;
      }
      if (containWD(grid + grid_idx, word,row,col)) {
        print_int(row);
        print_char(',');
        print_int(col);
        print_char(' ');
        print_char('D');
        print_char(' ');
        print_word(word);
        print_char('\n');
        a=1;
      }
    }
    grid_idx++;
    col++;
    }
    if(a==0){
      printf("%i",-1);
    }
  }
//---------------------------------------------------------------------------
// MAIN function
//---------------------------------------------------------------------------

int main (void)
{
  int dict_idx = 0;
  int start_idx = 0;
  /////////////Reading dictionary and grid files//////////////
  ///////////////Please DO NOT touch this part/////////////////
  int c_input;
  int idx = 0;


  // open grid file
  FILE *grid_file = fopen(grid_file_name, "r");
  // open dictionary file
  FILE *dictionary_file = fopen(dictionary_file_name, "r");

  // if opening the grid file failed
  if(grid_file == NULL){
    printf("Error in opening grid file.\n");
    return -1;
  }

  // if opening the dictionary file failed
  if(dictionary_file == NULL){
    printf("Error in opening dictionary file.\n");
    return -1;
  }
  // reading the grid file
  do {
    c_input = fgetc(grid_file);
    // indicates the the of file
    if(feof(grid_file)) {
      grid[idx] = '\0';
      break;
    }
    grid[idx] = c_input;
    idx += 1;

  } while (1);

  // closing the grid file
  fclose(grid_file);
  idx = 0;

  // reading the dictionary file
  do {
    c_input = fgetc(dictionary_file);
    // indicates the end of file
    if(feof(dictionary_file)) {
      dictionary[idx] = '\0';
      break;
    }
    dictionary[idx] = c_input;
    idx += 1;
  } while (1);


  // closing the dictionary file
  fclose(dictionary_file);
  //////////////////////////End of reading////////////////////////
  ///////////////You can add your code here!//////////////////////
  idx=0;
  while(1){
     if(grid[idx]!='\n'){
       row_length++;
       idx++;
     }
     else break;
  }

  idx=0;
  while(1){
      if(grid[idx]=='\n'){
        gridheight++;
      }
      if(grid[idx]=='\0'){
        break;
      }
      idx++;
  }

  idx=0;
  while(1){
    if(grid[idx]=='\0'){
      break;
    }
    idx++;
    gridlength++;
  }

  idx = 0;
  do {
    c_input = dictionary[idx];
    if(c_input == '\0') {
      break;
    }
    if(c_input == '\n') {
      dictionary_idx[dict_idx ++] = start_idx;
      start_idx = idx + 1;
    }
    idx += 1;
  } while (1);

  dict_num_words = dict_idx;

  strfind();

  return 0;
}
