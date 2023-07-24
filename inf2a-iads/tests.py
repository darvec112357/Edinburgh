import math
import graph
import random
from itertools import permutations
 
def generate_euclid_graph(n,g):
    a=random.sample(range(1,n*20),n);##set the size of the window as twenty times the number of cities
    b=random.sample(range(1,n*20),n);
    c=[[a[i],b[i]]for i in range (n)];
    g.n=n;
    g.dist=[[0 for i in range (n)]for j in range (n)];
    for i in range (n):
        for j in range (n):
            g.dist[i][j]=graph.euclid(c[i],c[j]);
    g.perm=[i for i in range (n)];

    
#this function find the union of all given intervals, if the union does not exist, return the empty set
def min_inter(a):
    l=len(a);
    ma=max([a[i][0] for i in range (l)]);
    mi=min([a[i][1] for i in range (l)]);
    if(ma<mi):
       return [ma,mi];
    else:
       return [];

def generate_metric_graph(n,g):
    g.n=n;
    g.dist=[[99999 for i in range (n)]for j in range (n)];
    for i in range (n):
        for j in range (i+1):
            if(j==i):
                g.dist[i][j]=0;
            else:
                ##for another city k that is different from i and j, we first find all the intervals for each possible
                ##triangle ijk that satisfy the triangle inequality, then take the union of the interval, which will
                ##give the interval that the weight edge ij should take
                a=[[abs(g.dist[k][i]-g.dist[k][j]),(g.dist[k][i]+g.dist[k][j])]for k in range (n) if (k!=i and k!=j) ];           
                inter=min_inter(a);               
                g.dist[i][j]=random.randint(int(inter[0]),int(inter[1]));
                g.dist[j][i]=g.dist[i][j];
            


def brute_force(g):
    min_tour=math.inf;
    n=g.n;
    permu=permutations([i for i in range (n)]);
    for i in list(permu):
        g.perm=[i[j] for j in range (n)];
        if(g.tourValue()<min_tour):
            min_tour=g.tourValue();
    return min_tour;

def final_test_euclid_swap_TwoOpt(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_euclid_graph(size,g);
        result=brute_force(g);
        g.swapHeuristic();
        g.TwoOptHeuristic();
        approx_swap_TwoOpt=g.tourValue();
        accuracy=result/approx_swap_TwoOpt;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def final_test_euclid_greedy(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+":");
        generate_euclid_graph(size,g);
        result=brute_force(g);
        g.Greedy();
        approx_greedy=g.tourValue();
        accuracy=result/approx_greedy;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def final_test_euclid_insertion(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+":");
        generate_euclid_graph(size,g);
        result=brute_force(g);
        g.insertion_Heuristic();
        approx_insertion=g.tourValue();
        accuracy=result/approx_insertion;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def final_test_metric_swap_TwoOpt(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_metric_graph(size,g);
        result=brute_force(g);
        g.swapHeuristic();
        g.TwoOptHeuristic();
        approx_swap_TwoOpt=g.tourValue();
        accuracy=result/approx_swap_TwoOpt;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def final_test_metric_greedy(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_metric_graph(size,g);
        result=brute_force(g);
        g.Greedy();
        approx_greedy=g.tourValue();
        accuracy=result/approx_greedy;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def final_test_metric_insertion(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_metric_graph(size,g);
        result=brute_force(g);
        g.insertion_Heuristic();
        approx_insertion=g.tourValue();
        accuracy=result/approx_insertion;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def random_split(n):
    a=random.sample(range(1,n),3);
    a.sort();
    p0=a[0];
    p1=a[1]-a[0];
    p2=a[2]-a[1];
    p3=n-a[2];
    array0=[i for i in range (p0)];
    array1=[i+a[0] for i in range (p1)];
    array2=[i+a[1] for i in range (p2)];
    array3=[i+a[2] for i in range (p3)];
    answer=[array0,array1,array2,array3];
    return answer;

#construct a square such that the best solution is given by its perimeter
def generate_random_graph(n,g):
    g.n=n;
    array=random_split(n-4);
    #set the side length to be 10*n
    s1=random.sample(range(1,n*10),len(array[0]));
    s2=random.sample(range(1,n*10),len(array[1]));
    s3=random.sample(range(1,n*10),len(array[2]));
    s4=random.sample(range(1,n*10),len(array[3]));
    coordinates1=[[s1[i],0]for i in range (len(s1))];
    coordinates2=[[n*10,s2[i]]for i in range (len(s2))];
    coordinates3=[[s3[i],n*10]for i in range (len(s3))];
    coordinates4=[[0,s4[i]]for i in range (len(s4))];
    coordinates=[[0,0]]+coordinates1+[[n*10,0]]+coordinates2+[[n*10,n*10]]+coordinates3+[[0,n*10]]+coordinates4; 
    g.dist=[[graph.euclid(coordinates[i],coordinates[j]) for i in range (n)]for j in range (n)];
    g.perm=[i for i in range (n)];
    print(g.tourValue());
                    
def test_large_input_swap_TwoOpt(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_random_graph(size,g);
        result=40*size;
        g.perm=[i for i in range (size)];
        g.swapHeuristic();
        g.TwoOptHeuristic();
        approx_swap_TwoOpt=g.tourValue();
        accuracy=result/approx_swap_TwoOpt;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def test_large_input_greedy(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_random_graph(size,g);
        result=40*size;
        g.perm=[i for i in range (size)];
        g.Greedy();
        approx_greedy=g.tourValue();
        accuracy=result/approx_greedy;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);

def test_large_input_insertion(n,g,size):
    total_accuracy=0;
    for i in range (1,n+1):
        #print("Iteration "+str(i)+";");
        generate_random_graph(size,g);
        result=40*size;
        g.perm=[i for i in range (size)];
        g.insertion_Heuristic();
        approx_insertion=g.tourValue();
        accuracy=result/approx_insertion;
        total_accuracy=total_accuracy+accuracy;
        #print("Accuracy of estimation is "+str(accuracy)+"\n");
    print(total_accuracy/n);



    
    
    
