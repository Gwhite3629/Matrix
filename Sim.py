from Types import Matrix as M
from decimal import *

A = M(3, 2)
B = M(2, 3)

A.data = [[5,4],[7,6],[9,8]]
B.data = [[9,8,5],[7,6,4]]

E = A.product(B)

print(f'A.product(B): {E.data}')

print(f'A Original:{A.data}')
A.T()
print(f'A Transpose:{A.data}, A.Shape:{A.shape()}')
A.T()
print(f'A Original: {A.data}, A.Shape: {A.shape()}')
A.reshape(3, 3)
print(f'A Resized: {A.data}, A.Shape {A.shape()}')

print(f'B Original: {B.data}')
D = B.copy()
print(f'D Copy of B: {D.data}')
B.reshape(3, 3)
print(f'B Resized: {B.data}')

C = A.product(B)

print(f'A.Product(B): {C.data}')

T = M(3, 3)

T.data = [[2, 3, 1],
          [4, 5, 6],
          [7, 8, 9]]

print(f'Rank of T: {T.rank()}')

B = T.invert()

B.round(10)

D = T.det()

print(f'Inverse of T: {B.data}')
print(f'Determinant of T: {D}')

U = M(3, 3)

U.data = [[12, -51, 4],
          [6, 167, -68],
          [-4, 24, -41]]

(Q, R) = U.QR()

QT = Q.copy()

QT.T()

I = Q.product(QT)

Q.round(10)
R.round(10)
I.round(10)

print(f'U: {U.data}')
print(f'Q of U: {Q.data}')
print(f'R of U: {R.data}')
print(f'Identity: {I.data}')