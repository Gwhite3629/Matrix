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
        if self.rows == 1:
            for col in range(self.cols):
                self.data[col] = self.data[col] * mul
        else :
            for col in range(self.cols):
                self.data[row][col] = self.data[row][col] * mul

    def Rowadd(self, row1: Iterable, row2: Iterable, mul: float) -> None:
        VecT = Matrix(1, self.cols)
        VecT.data = self.data[row2]
        VecT.Rowmul(0, mul)
        for col in range(self.cols):
            self.data[row1][col] = self.data[row1][col] + VecT.data[col]

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

    def echelon(self) -> None:
        h = 0
        k = 0

        while h<self.rows and k<self.cols:
            imax = self.argmax(col=1, index=(h,k))
            if self.data[imax][k] == 0:
                k = k + 1
            else:
                self.swap(h, imax)
                for i in range(h+1, self.rows):
                    f = self.data[i][k]/self.data[h][k]
                    self.data[i][k] = 0
                    for j in range(k+1, self.cols):
                        self.data[i][j] = self.data[i][j] - self.data[h][j] * f
                h = h + 1
                k = k + 1

    def reduce(self, track=0):
        lead = 0
        d = 1

        for r in range(self.rows):
            if lead >= self.cols:
                return d
            i = r
            while self.data[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.cols == lead:
                        return d
            self.data[i],self.data[r] = self.data[r],self.data[i]
            d *= -1
            lv = self.data[r][lead]
            self.data[r] = [ mrx / float(lv) for mrx in self.data[r]]
            d *= 1/float(lv)
            for i in range(self.rows):
                if i != r:
                    lv = self.data[i][lead]
                    self.data[i] = [iv - lv*rv for rv,iv in zip(self.data[r],self.data[i])]
            lead += 1
        if (track):
            return d

    def argmax(self, row=0, col=0, index=(0,0)):
        max = 0

        if row == col == 0:
            for i in range(index[0], self.rows):
                for j in range(index[1], self.cols):
                    if abs(self.data[i][j]) > max:
                        max = self.data[i][j]
                        ind = (i, j)
        elif row == 0 and col != 0:
            for i in range(index[0], self.rows):
                if abs(self.data[i][index[1]]) > max:
                    max = self.data[i][index[1]]
                    ind = i
        elif row != 0 and col == 0:
            for j in range(index[1], self.rows):
                if abs(self.data[index[0]][j]) > max:
                    max = self.data[index[0]][j]
                    ind = j
        return ind

    def augment(self, Mat: 'Matrix') -> None:
        if self.rows != Mat.rows:
            print("Matrices must have same number of rows")
            return IndexError
        self.reshape(self.rows, self.cols+Mat.cols)
        if Mat.cols == 1:
            for i in range(self.rows):
                self.data[i][self.cols] = Mat.data[i]
        else:
            for i in range(self.rows):
                for j in range(self.cols-Mat.cols, self.cols):
                    self.data[i][j] = Mat.data[i][j-self.cols]

    def invert(self) -> 'Matrix':
        if (self.rows != self.cols):
            print("Matrix must be square")
            return IndexError
        I = Matrix(self.rows, self.cols)
        I.Identity()
        copy = self.copy()
        copy.augment(I)
        copy.reduce()
        Test = copy.slice((0,copy.rows-1),(0,copy.rows-1))
        if (Test.compare(I)):
            inv = copy.slice((0,copy.rows-1),(copy.rows,2*copy.rows-1))
            return inv
        else:
            print("Inverse Matrix does not exist")
            return I

    def slice(self, rows, cols) -> 'Matrix':
        Out = Matrix(rows[1]-rows[0]+1,cols[1]-cols[0]+1)
        for i in range(Out.rows):
            for j in range(Out.cols):
                Out.data[i][j] = self.data[rows[0]+i][cols[0]+j]
        return Out

    def Identity(self) -> None:
        if self.rows != self.cols:
            self.reshape(self.rows, self.rows)
        for i in range(self.rows):
            self.data[i][i] = 1

    def compare(self, Mat: 'Matrix') -> bool:
        if (self.rows != Mat.rows or self.cols != Mat.cols):
            print("Error matrices must be the same size")
        for i in range(self.rows):
            for j in range(self.cols):
                if (math.isclose(self.data[i][j],Mat.data[i][j],rel_tol=1e-3) != 1):
                    return 0
        return 1

#    def norm (self) -> Matrix:

    def det (self) -> float:
        R = self.copy()
        d = R.reduce(track=1)
        print(d)
        D = 1/d
        B = self.invert()
        for i in range(B.rows):
            D = D * B.data[i][i]
            print(D)
        return D