import math

def euclid(p,q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)

def convert(a):
    b=a.strip();
    c=b.split(' ');
    answer=[int(c[0]),int(c[-1])];
    return answer;

def convert2(a):
    b=a.strip();
    c=b.split(' ');
    answer=[int(c[0]),int(c[1]),int(c[2])];
    return answer;

def contain(a,b):
    for i in range (len(b)):
        if(a==b[i]):
             return True;
    return False;


class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self,n,filename):
        if(n==-1):
            lineList=[line.rstrip('\n') for line in open(filename)];           
            self.n=len(lineList);
            coordinates=[[0 for j in range (2)]for i in range (self.n)];
            for i in range (self.n):
                coordinates[i]=convert(lineList[i]);
            self.dist=[[0 for j in range (self.n)]for i in range (self.n)];
            for i in range (self.n):
                for j in range (self.n):
                    self.dist[i][j]=euclid(coordinates[i],coordinates[j]);           
            self.perm=[i for i in range (self.n)];
        else:
            lineList=[line.rstrip('\n') for line in open(filename)];
            l=len(lineList);         
            self.n=n;
            arr=[[0 for j in range (3)]for i in range (l)];
            for i in range (l):
                arr[i]=convert2(lineList[i]);
            self.dist=[[0 for i in range (n)]for j in range (n)];
            for i in range (n):
                for j in range (n):
                    for k in range (l):
                        if((i==arr[k][0] and j==arr[k][1])or(i==arr[k][1] and j==arr[k][0])):
                            self.dist[i][j]=arr[k][2];          
            self.perm=[i for i in range (n)];   
            
            
    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        answer=0;
        l=self.n;
        for i in range (l):
            if(i!=l-1):
                answer+=self.dist[self.perm[i]][self.perm[i+1]];
            else:
                answer+=self.dist[self.perm[l-1]][self.perm[0]];
        return answer;
            
    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self,i):
        n=self.n;
        pre=self.dist[self.perm[i]][self.perm[(i-1)%n]]+self.dist[self.perm[(i+1)%n]][self.perm[(i+2)%n]];
        new=self.dist[self.perm[(i-1)%n]][self.perm[(i+1)%n]]+self.dist[self.perm[i]][self.perm[(i+2)%n]];
        if(new<pre):
            a=self.perm[i];
            self.perm[i]=self.perm[(i+1)%n];
            self.perm[(i+1)%n]=a;
            return True;
        else:
            return False;

    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.
    def reverse(self,start,end):
        b=[self.perm[i] for i in range (self.n)];    
        d=end-start;
        for i in range (d+1):
           self.perm[start+i]=b[end-i];
        return self.perm;
        
    def tryReverse(self,i,j):
        if(i==j):
            return True;
        n=self.n;
        pre=self.dist[self.perm[i]][self.perm[(i-1)%n]]+self.dist[self.perm[j]][self.perm[(j+1)%n]];
        new=self.dist[self.perm[j]][self.perm[(i-1)%n]]+self.dist[self.perm[i]][self.perm[(j+1)%n]];
        if(new<pre):
            self.reverse(i,j);
            return True;
        else:
            return False;
        

    def swapHeuristic(self):
        better = True
        while better:
            better = False
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self):
        better = True
        while better:
            better = False
            for j in range(self.n-1):
                for i in range(j):
                    if self.tryReverse(i,j):
                        better = True
                

    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.

    
    def Greedy(self):
        answer=[0 for i in range (self.n)];
        for i in range (self.n-1):
            min_dist=math.inf;
            min_idx=0;
            for j in range (self.n):         
                if(not(j in answer) and self.dist[answer[i]][j]<min_dist):
                    min_dist=self.dist[answer[i]][j];
                    min_idx=j;           
            answer[i+1]=min_idx;                
        self.perm=answer;

    def part_tourValue(self,a):
        answer=0;
        for i in range (len(a)):
            if(i!=len(a)-1):
                answer=answer+self.dist[a[i]][a[i+1]];
            answer=answer+self.dist[a[len(a)-1]][a[0]];
        return answer;
    
    def insertion_Heuristic(self):
        n=self.n;
        answer=[0];
        permu=[i for i in range (n)];
        permu.remove(0);
        while(permu):
            min_dist=math.inf;
            min_pos=0;
            min_idx=0;
            for i in range (n):
                for j in range (len(answer)+1):
                    if(not(i in answer)):
                        answer.insert(j,i);
                        value=self.part_tourValue(answer);
                        if(value<min_dist):
                            min_dist=value;
                            min_pos=j;
                            min_idx=i;
                        answer.remove(i);
            answer.insert(min_pos,min_idx);
            permu.remove(min_idx);
        self.perm=answer;
        
    
        















            
            
        
