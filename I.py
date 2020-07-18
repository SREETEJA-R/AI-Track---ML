
from scipy.optimize import linprog
object=[-50,-120]
left_para=[[7000,2000],[10,30],[1,1]]
right_para=[700000,1200,110]
boun=[(0,float("inf")),(0,float("inf"))]
opt=linprog(c=object,A_ub=left_para,b_ub=right_para,bounds=boun,method="simplex")
opt


