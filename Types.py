import math
from typing import Iterable

class Matrix(object):
    def __init__(self,rows: int,cols: int) -> None:
        self.data = [[float] * cols for _ in range(rows)]
        if (rows == 0 | cols == 0):
            self.rows = 1
            self.cols = 1
        else:
            self.rows = rows
            self.cols = cols
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = 0.0

    def ins(self,row: Iterable,col: Iterable,datum: float) -> None:
        if (row >= self.rows or col >= self.cols):
            print("Resizing")
            self.reshape(row, col)
        self.data[row][col] = datum

    def reshape(self,rows: int,cols: int) -> None:
        new = Matrix(rows, cols)
        for row in range(self.rows):
            for col in range(self.cols):
                try:
                    new.data[row][col] = self.data[row][col]
                except:
                    pass
        self.data = new.data
        self.rows, self.cols = new.rows, new.cols

    def add(self,Mat: 'Matrix') -> 'Matrix':
        if (self.rows != Mat.rows or self.cols != Mat.cols):
            print("Matrices must have same number of rows and cols")
            return BufferError
        Out = Matrix(self.rows, self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                Out.data[row][col] = self.data[row][col] + Mat.data[row][col]

        return Out

    def Scamul(self,Scalar: float) -> 'Matrix':
        Out = Matrix(self.rows, self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                Out.data[row][col] = self.data[row][col] * Scalar

        return Out

    def product(self,Mat: 'Matrix') -> 'Matrix':
        if (self.cols != Mat.rows):
            print("Columns of matrix A must equal rows of matrix B")
            return BufferError
        Out = Matrix(self.rows, Mat.cols)
        for i in range(self.rows):
            for j in range(Mat.cols):
                for n in range(self.cols):
                    Out.data[i][j] += (self.data[i][n] * Mat.data[n][j])
        return Out

    def dot(self,Mat: 'Matrix') -> float:
        if ((self.rows != 1 and Mat.rows != 1) or (self.cols != Mat.cols)):
            print("Inputs must be vectors of the same size")
            return IndexError
        Scalar = 0
        for row in range(self.rows):
            Scalar += self.data[row][0] * Mat.data[row][0]
        return Scalar

    def trace(self) -> float:
        Scalar = 0
        if (self.rows != self.cols):
            print("Matrix must be square")
            return IndexError
        for row in range(self.rows):
            Scalar = Scalar + self.data[row][row]

        return Scalar

    def swap(self,row1: Iterable,row2: Iterable) -> None:
        VecT = Matrix(1, self.rows)
        VecT.data[0] = self.data[row1]
        self.data[row1] = self.data[row2]
        self.data[row2] = VecT.data[0]
    
    def Rowmul(self,row: Iterable,mul: float) -> None:
        for col in range(self.cols):
            self.data[row][col] = self.data[row][col] * mul

    def Rowadd(self,row: Iterable,mul: float) -> None:
        VecT = Matrix(1, self.rows)
        VecT = self.data[row] * mul
        for col in range(self.cols):
            self.data[row][col] = self.data[row][col] + VecT

    def copy(self) -> 'Matrix':
        copy = Matrix(self.rows, self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                copy.data[row][col] = self.data[row][col]

        return copy

    def T(self) -> None:
        new = Matrix(rows=self.cols, cols=self.rows)
        for row in range(self.rows):
            for col in range(self.cols):
                new.data[col][row] = self.data[row][col]
        self.data = new.data
        self.rows, self.cols = new.rows, new.cols

    def shape(self) -> tuple:
        return self.rows, self.cols

    def normalize(self) -> None:
        if ((self.rows != 1) or (self.cols != 1)):
            print("Input must be vector")
            return IndexError
        mag = self.vecnorm
        if (self.cols != 1):
            for col in self.cols:
                self.data[0][col] = self.data[0][col] / mag
        else:
            for row in self.rows:
                self.data[row][0] = self.data[row][0] / mag

    def vecnorm(self) -> float:
        if ((self.rows != 1) or (self.cols != 1)):
            print("Input must be vector")
            return IndexError
        mag = 0
        if (self.cols != 1):
            for col in self.cols:
                mag = mag + self.data[0][col]
        else:
            for row in self.rows:
                mag = mag + self.data[row][0]
        mag = math.sqrt(mag)
        return mag

#    def norm (self) -> Matrix:

#    def det (self) -> float: