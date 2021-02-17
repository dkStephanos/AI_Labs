import os
import random
import sys
import time

RATE = 0.1
CLEAR = "CLS"     # "CLS" for Windows
queens = {}         # Dictionary showing the location of each queen
size = 10 # Get the board size from the command line

board = [[0 for c in range(size)] for r in range(size)]

def display(pBoard):
    for r in range(size):
        line = " "
        for c in range(size):
            st = ""
            if pBoard[r][c] == 0:
                st = str(pBoard[r][c])
            else:
                st = str(pBoard[r][c])
            line += st + " "
        print(line)


def addToThreats(row, col, change):

    for j in range(0,row):
        board[j][col] += change

    for j in range(row+1,size):
        board[j][col] += change
        if ( col+(j-row) < size ):
            board[j][col+(j-row)] += change
        if( col-(j-row) >= 0):
            board[j][col-(j-row)] += change


def backtrack_search(row=0):
    if row == size:
        os.system(CLEAR)
        print("SOLUTION:\n")
        display(board)
        return True
    else:
        for col in range(0,size):
            queens[row] = col
            if board[row][col] == 0:
                addToThreats(row,col,1)
                os.system(CLEAR)
                print("Row",row,"\n")
                display(board)
                time.sleep(RATE)
                status = backtrack_search(row+1)
                if (status):
                    return True
                addToThreats(row,col,-1)    # BACKTRACK
                os.system(CLEAR)
                print("Row",row,"\n")
                display(board)
                time.sleep(RATE)
    return False

if (not backtrack_search()):
    print("NO SOLUTION!")
