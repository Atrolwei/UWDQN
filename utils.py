# below is the code of points to a xcoded number    
def dot2num(point,grid_size):
    return point[0] * grid_size + point[1]

def to_xcoded_number(points,grid_size,xcimal=16):
    xcoded_number=0
    len_points=len(points)
    for idx,point in enumerate(points):
        xcoded_number+=dot2num(point,grid_size)*xcimal**(len_points-idx-1)
    return xcoded_number

# below is the code of xcoded number to points
def num2dot(num,grid_size):
    return (num//grid_size,num%grid_size)

def to_points(xcoded_number,n_points,grid_size,xcimal=16):
    points=[(0,0)]*n_points
    idx=1
    while xcoded_number:
        points[-idx]=num2dot(xcoded_number%xcimal,grid_size)
        xcoded_number//=xcimal
        idx+=1
    return points

def num2acts(num,n_acts,xcimal=4):
    acts=[0]*n_acts
    idx=1
    while num:
        acts[-idx]=num%xcimal
        num//=xcimal
        idx+=1
    return acts

def acts2num(acts,xcimal=5):
    num=0
    acts=acts[::-1]
    for idx,act in enumerate(acts):
        num=num+act*xcimal**idx
    return num



