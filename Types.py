import math
from pickle import TRUE
from typing import Iterable
from decimal import *

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
                self[row][col] = 0.0

    def __str__(self):
        rounded = self.copy()
        rounded.round(10)
        return ('|\n'.join('|' + '  '.join(map(str, r)) for r in rounded)) + '|'

    def __getitem__(self,index):
        return self.data[index]

    def __setitem__(self, index, val):
        self.data[index] = val

    def ins(self,row: Iterable,col: Iterable,datum: float) -> None:
        if (row >= self.rows or col >= self.cols):
            print("Resizing")
            self.reshape(row, col)
        self[row][col] = datum

    def reshape(self,rows: int,cols: int) -> None:
        new = Matrix(rows, cols)
        for row in range(self.rows):
            for col in range(self.cols):
                try:
                    new[row][col] = self[row][col]
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
                Out[row][col] = self[row][col] + Mat[row][col]

        return Out

    def Scamul(self,Scalar: float) -> 'Matrix':
        Out = Matrix(self.rows, self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                Out[row][col] = self[row][col] * Scalar

        return Out

    def product(self,Mat: 'Matrix') -> 'Matrix':
        if (self.cols != Mat.rows):
            print("Columns of matrix A must equal rows of matrix B")
            return BufferError
        Out = Matrix(self.rows, Mat.cols)
        if self.rows == 1:
            for i in range(self.cols):
                for j in range(Mat.rows):
                    Out[0][0] += (self[i] * Mat[j])
        elif self.cols == 1:
            for i in range(self.rows):
                for j in range(Mat.cols):
                    Out[0][0] += (self[i] * Mat[j])
        else:
            Out = Matrix(self.rows, Mat.cols)
            for i in range(self.rows):
                for j in range(Mat.cols):
                    for n in range(self.cols):
                        Out[i][j] += (self[i][n] * Mat[n][j])
        return Out

    def dot(self,Mat: 'Matrix') -> float:
        if ((self.rows != 1 and Mat.rows != 1) or (self.cols != Mat.cols)):
            print("Inputs must be vectors of the same size")
            return IndexError
        Scalar = 0
        for row in range(self.rows):
            Scalar += self[row][0] * Mat[row][0]
        return Scalar

    def trace(self) -> float:
        Scalar = 0
        if (self.rows != self.cols):
            print("Matrix must be square")
            return IndexError
        for row in range(self.rows):
            Scalar = Scalar + self[row][row]

        return Scalar

    def swap(self,row1: Iterable,row2: Iterable) -> None:
        VecT = Matrix(1, self.rows)
        VecT[0] = self[row1]
        self[row1] = self[row2]
        self[row2] = VecT[0]
    
    def Rowmul(self,row: Iterable,mul: float) -> None:
        if self.rows == 1:
            for col in range(self.cols):
                self[col] = self[col] * mul
        else :
            for col in range(self.cols):
                self[row][col] = self[row][col] * mul

    def Rowadd(self, row1: Iterable, row2: Iterable, mul: float) -> None:
        VecT = Matrix(1, self.cols)
        VecT.data = self[row2]
        VecT.Rowmul(0, mul)
        for col in range(self.cols):
            self[row1][col] = self[row1][col] + VecT[col]

    def Eadd(self, row: Iterable, val: float) -> None:
        for i in range(self.cols):
            self[row][i] = self[row][i] + val

    def copy(self) -> 'Matrix':
        copy = Matrix(self.rows, self.cols)
        if (self.rows == 1):
            copy.data = self.data
        elif (self.cols == 1):
            for j in range(self.cols):
                copy[j] = self[j]
        else:
            for row in range(self.rows):
                for col in range(self.cols):
                    copy[row][col] = self[row][col]

        return copy

    def T(self) -> None:
        new = Matrix(rows=self.cols, cols=self.rows)
        if self.rows == 1:
            for col in range(self.cols):
                new[col] = self[col]
        elif self.cols == 1:
            for row in range(self.rows):
                new[row] = self[row]
        else:
            for row in range(self.rows):
                for col in range(self.cols):
                    new[col][row] = self[row][col]
        self.data = new.data
        self.rows, self.cols = new.rows, new.cols

    def shape(self) -> tuple:
        return self.rows, self.cols

    def normalize(self) -> None:
        if ((self.rows != 1) and (self.cols != 1)):
            print("Input must be vector, norm")
            return IndexError
        mag = self.vecnorm()
        if (self.cols != 1):
            for col in range(self.cols):
                self[col] = self[col] / mag
        else:
            for row in range(self.rows):
                self[row] = self[row] / mag

    def vecnorm(self) -> float:
        if ((self.rows != 1) and (self.cols != 1)):
            print("Input must be vector, vecnorm")
            return IndexError
        mag = 0
        if (self.cols != 1):
            for col in range(self.cols):
                mag = mag + self[col]**2
        else:
            for row in range(self.rows):
                mag = mag + self[row]**2
        mag = math.sqrt(mag)
        return mag

    def echelon(self, track=0):
        h = 0
        k = 0
        d = 1

        while h<self.rows and k<self.cols:
            imax = self.argmax(col=1, index=(h,k))
            if self[imax][k] == 0:
                k = k + 1
            else:
                self.swap(h, imax)
                d *= -1
                for i in range(h+1, self.rows):
                    f = self[i][k]/self[h][k]
                    self[i][k] = 0
                    for j in range(k+1, self.cols):
                        self[i][j] = self[i][j] - self[h][j] * f
                h = h + 1
                k = k + 1
        if track:
            return d

    def echelonStrict(self):
        # TODO
        # Add echelon function that only
        # operates by adding multiples
        # of rows to others for simplification
        return

    def reduce(self, track=0):
        lead = 0
        d = 1

        for r in range(self.rows):
            if lead >= self.cols:
                return d
            i = r
            while self[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.cols == lead:
                        return d
            self[i],self[r] = self[r],self[i]
            d *= -1
            lv = self[r][lead]
            self[r] = [ mrx / float(lv) for mrx in self[r]]
            d *= 1/float(lv)
            for i in range(self.rows):
                if i != r:
                    lv = self[i][lead]
                    self[i] = [iv - lv*rv for rv,iv in zip(self[r],self[i])]
            lead += 1
        if (track):
            return d

    def argmax(self, row=0, col=0, index=(0,0)):
        max = 0

        if row == col == 0:
            for i in range(index[0], self.rows):
                for j in range(index[1], self.cols):
                    if abs(self[i][j]) > max:
                        max = self[i][j]
                        ind = (i, j)
        elif row == 0 and col != 0:
            for i in range(index[0], self.rows):
                if abs(self[i][index[1]]) > max:
                    max = self[i][index[1]]
                    ind = i
        elif row != 0 and col == 0:
            for j in range(index[1], self.rows):
                if abs(self[index[0]][j]) > max:
                    max = self[index[0]][j]
                    ind = j
        return ind

    def argmin(self, row=0, col=0, index=(0,0)):
        min = 0

        if row == col == 0:
            for i in range(index[0], self.rows):
                for j in range(index[1], self.cols):
                    if abs(self[i][j]) < min:
                        min = self[i][j]
                        ind = (i, j)
        elif row == 0 and col != 0:
            for i in range(index[0], self.rows):
                if abs(self[i][index[1]]) < min:
                    min = self[i][index[1]]
                    ind = i
        elif row != 0 and col == 0:
            for j in range(index[1], self.rows):
                if abs(self[index[0]][j]) < min:
                    min = self[index[0]][j]
                    ind = j
        return ind

    def augment(self, Mat: 'Matrix') -> None:
        if self.rows != Mat.rows:
            print("Matrices must have same number of rows")
            return IndexError
        self.reshape(self.rows, self.cols+Mat.cols)
        if Mat.cols == 1:
            for i in range(self.rows):
                self[i][self.cols] = Mat[i]
        else:
            for i in range(self.rows):
                for j in range(self.cols-Mat.cols, self.cols):
                    self[i][j] = Mat[i][j-self.cols]

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
                Out[i][j] = self[rows[0]+i][cols[0]+j]
        return Out

    def Identity(self) -> None:
        if self.rows != self.cols:
            self.reshape(self.rows, self.rows)
        for i in range(self.rows):
            self[i][i] = 1

    def compare(self, Mat: 'Matrix') -> bool:
        if (self.rows != Mat.rows or self.cols != Mat.cols):
            print("Error matrices must be the same size")
        for i in range(self.rows):
            for j in range(self.cols):
                if (math.isclose(self[i][j],Mat[i][j],rel_tol=1e-3) != 1):
                    return 0
        return 1

#    def norm (self) -> Matrix:

    def det (self, method='echelon') -> float:
        match method:
            case 'echelon':
                U = self.copy()
                d = U.reduce(track=1)
                D = 1/d
                B = self.invert()
                for i in range(B.rows):
                    D = D * B[i][i]
                
                L = self.copy()
                d = L.echelon(track=1)
                D = D*d
                for i in range(B.rows):
                    D = D * 1/B[i][i]

                return D
            case 'laplace':
                D = 0
                for j in range(self.cols):
                    D += self[1][j] * self.cofactor(1, j)
                return D

    def rank(self) -> int:
        rank = 0
        E = self.copy()
        E.echelon()
        if E.rows > E.cols:
            E.T()
        for i in range(E.rows):
            for j in range(E.cols):
                if E[i][j] > 0:
                    rank += 1
                    break

        return rank

    def submatrix(self, row, col) -> 'Matrix':
        R = self.delrow(row)
        sub = R.delcol(col)

        return sub

    def delrow(self, row) -> 'Matrix':
        aug = Matrix(self.rows-1, self.cols)
        flag = 0
        for i in range(aug.rows):
            for j in range(aug.cols):
                if ((i == (row)) | flag):
                    aug[i][j] = self[i+1][j]
                    flag = 1
                else:
                    aug[i][j] = self[i][j]

        return aug

    def delcol(self, col) -> 'Matrix':
        aug = Matrix(self.rows, self.cols-1)
        flag = 0
        for i in range(aug.rows):
            for j in range(aug.cols):
                if((j == (col)) | flag):
                    aug[i][j] = self[i][j+1]
                    flag = 1
                else:
                    aug[i][j] = self[i][j]

        return aug

    def minor(self, row, col) -> float:
        sub = self.submatrix(row, col)
        det = sub.det()
        return det

    def cofactor(self, row, col) -> float:
        det = self.minor(row, col)
        cofactor = det*(-1)**(row+col)
        return cofactor

#    def dim(self) ->
#    def span(self) -> 
#    def basis(self) ->
#    def LU(self) ->
#    def perm(self) ->
#    def CT(self) ->
#    def eigenvalue(self) ->
#    def eigenvector(self, EV) ->

    def QR(self) -> tuple['Matrix', 'Matrix']:
        n = self.cols
        m = self.rows
        A = self.copy()
        Q = Matrix(m, n)
        R = Matrix(n, n)
        for k in range(n):
            s = 0
            for j in range(m):
                s += (A[j][k])**2
            R[k][k] = math.sqrt(s)
            for j in range(m):
                Q[j][k] = (A[j][k])/(R[k][k])
            for i in range(k+1,n):
                s = 0
                for j in range(m):
                    s += (A[j][i]) * (Q[j][k])
                R[k][i] = s
                for j in range(m):
                    A[j][i] -= (R[k][i] * Q[j][k])
        return (Q, R)

    def orthonormal(self) -> 'Matrix':
        self.T()
        U = Matrix(self.rows,self.cols)
        V = Matrix(1,self.cols)
        V.data = self[0]
        V = V.copy()
        V.normalize()
        U[0] = V.data
        for i in range(1,self.cols):
            U[i] = self[i]
            for j in range(0,i-1):
                V1, V2 = Matrix(1,self.cols), Matrix(1, self.cols)
                V1.data = U[j]
                V2.data = U[i]
                print(V2.shape())
                V2.T()
                print(V2.shape())
                n = V1.product(V2)
                N = n[0][0]
                D = (V1.vecnorm())**2
                val = -1*N/D
                U.Rowadd(i,j,val)
            Urow = Matrix(1,self.cols)
            Urow.data = U[i]
            Urow.normalize()
            U[i] = Urow.data
        self.T()
        U.T()
        return U

    def gram(self) -> 'Matrix':
        T = self.copy()
        T.T()
        G = self.product(T)
        return G

    def round(self, precision) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                self[i][j] = round(self[i][j], precision)