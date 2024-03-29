from Types import Matrix as M
from decimal import *

A = M(3, 2)
B = M(2, 3)

A.data = [[5,4],[7,6],[9,8]]
B.data = [[9,8,5],[7,6,4]]

E = A.product(B)

print(f'A.product(B):\n{E}')

print(f'A Original:\n{A}')
A.T()
print(f'A Transpose:\n{A}, A.Shape:{A.shape()}')
A.T()
print(f'A Original:\n{A}, A.Shape: {A.shape()}')
A.reshape(3, 3)
print(f'A Resized:\n{A}, A.Shape {A.shape()}')

print(f'B Original:\n{B}')
D = B.copy()
print(f'D Copy of B:\n{D}')
B.reshape(3, 3)
print(f'B Resized:\n{B}')

C = A.product(B)

print(f'A.Product(B):\n{C}')

T = M(3, 3)

T.data = [[2, 3, 1],
          [4, 5, 6],
          [7, 8, 9]]

print(f'Rank of T: {T.rank()}')

B = T.invert()

B.round(10)

D = T.det()

print(f'Inverse of T:\n{B}')
print(f'Determinant of T: {D}')

U = M(3, 3)

U.data = [[2, 4, 7],
          [3, 1, 5],
          [-1, 9, 21]]

print(f'U: \n{U}')

(Q, R) = U.QR()

print(f'Q: \n{Q}')
print(f'R: \n{R}')
print(f'P: \n{Q.product(R)}')

print(f'Echelon Det(U): {U.det()}')
print(f'Laplace Det(U): {U.det(method="laplace")}')

C = M(3, 3)

C.data = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

cof = M(3,3)

cof.data = [[C.cofactor(0, 0),C.cofactor(0, 1),C.cofactor(0, 2)],
            [C.cofactor(1, 0),C.cofactor(1, 1),C.cofactor(1, 2)],
            [C.cofactor(2, 0),C.cofactor(2, 1),C.cofactor(2, 2)]]

print(f'Laplace Det(C): {C.det(method="laplace")}')

print(f'Cofactor of C: \n{cof}')

(Q, R) = C.QR()

print(f'Q: \n{Q}')
print(f'R: \n{R}')
print(f'P: \n{Q.product(R)}')

A = M(3, 3)
B = M(3, 3)

A.data = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
B.data = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

print(A.product(B))

A.data = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

print(A.eigenvalues(1000))

#print(f'E: {A.det()}')
print(f'L: {A.det("laplace")}')
print(f'QR: {A.det("QR")}')
print(f'E: {A.det("pseudo")}')

L = M(5,5)

L.data = [[1, 2, 3, 4, 0],
          [5, 4, 0, 0, 1],
          [2, 0, 0, 4, 0],
          [3, 4, 2, 0, 1],
          [2, 0, 5, 4, 3]]

print(L.det())
print(L.eigenvalues(1000))