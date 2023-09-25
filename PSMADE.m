function [PSMADE_value,PSMADE_position,PSMADE_curve]=PSMADE(n,Max_iteration,lb,ub,dim,fobj)
%   [xo,Ot,nS]=powell(S,x0,ip,method,Lb,Ub,problem,tol,mxit)
%   n: Search slime moulds number
%   Max_iteration: The maximum number of iterations.
%   lb: Lower boundary
%   ub: Upper boundary
%   dim: The dimensionality of the problem
%   fobj: Objective function
%   PSMADE_position: The destination position
%   PSMADE_curve: The fitness value of the best solution found at each iteration
disp('PSMADE is now tackling your problem')
t = cputime;
% Initialize position
PSMADE_position = zeros(1,dim);
PSMADE_value = inf;  %Change this to -inf for maximization problems
all_fitness = inf*ones(n,1);  %Record the fitness of all slime moulds
weight = ones(n,dim);  %Fitness weight of each slime mould

%Initialize the set of random solutions
x = initialization(n,dim,ub,lb);  
PSMADE_curve=[];  

it = 1;  %Iteration count
lb=ones(1,dim).*lb; % Lower boundary
ub=ones(1,dim).*ub; % Upper boundary
z=0.03; % Parameter z=0.03
beta_min=0.4; % Lower bound of scaling factor
beta_max=1; % Upper bound of scaling factor
pb=zeros(n,dim);  %The optimal solution found by each individual
pb_fit=inf*ones(n,1);  %Fitness of the optimal solution found by each individual
taboo=[];  %Taboo table


% Main loop
while  it < Max_iteration
    %sort the fitness
    for i=1:n
        % Check if solutions go outside the search space and bring them back
        Flag4ub=x(i,:)>ub;
        Flag4lb=x(i,:)<lb;
        x(i,:)=(x(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        all_fitness(i) = fobj(x(i,:));

        if all_fitness(i) < PSMADE_value 
            PSMADE_value = all_fitness(i);
            PSMADE_position = x(i,:);
        end
        if all_fitness(i)<pb_fit(i)
            pb(i,:)=x(i,:);
            pb_fit(i)=all_fitness(i);
        end
    end
     
    PSMADE_curve(it) = PSMADE_value;
    it = it + 1;
    
    %% DE mechanism
    [x,pb,all_fitness,pb_fit] = funcDE(fobj,x,pb,all_fitness,pb_fit,lb,ub,beta_min,beta_max);
    %%
    
    [smell_order,smell_index] = sort(all_fitness);  
    worst_fitness = smell_order(n);
    best_fitness = smell_order(1);

    S=best_fitness-worst_fitness+eps;  % plus eps to avoid denominator zero
    
     %update the best fitness value and best position
    if best_fitness < PSMADE_value
        PSMADE_position=x(smell_index(1),:);
        PSMADE_value = best_fitness;
    end

    %calculate the fitness weight of each slime mold
    for i=1:n
        for j=1:dim
            if i<=(n/2)    %Eq.(2)
                weight(smell_index(i),j) = 1+rand()*log10((best_fitness-smell_order(i))/(S)+1);
            else
                weight(smell_index(i),j) = 1-rand()*log10((best_fitness-smell_order(i))/(S)+1);
            end
        end
    end
    
    %% Powell mechanism
    if it>0.8*Max_iteration
        % Check if PSMADE_position is in the taboo table
        [M,~] = size(taboo);  
        flag = 0;
        for m = 1:M
            if PSMADE_position == taboo(m,1) & PSMADE_value == taboo(m,2)
                flag = 1;
            end
        end
        
        if flag==0  % no
            % Add PSMADE_position to the taboo table
            taboo = [taboo;[PSMADE_position,PSMADE_value]];
            
            [x_powell,xp_fitness]=powell(fobj,PSMADE_position',0,1,lb,ub,-1,1e-6);
            x_powell = x_powell';
            if xp_fitness<PSMADE_value  % Eq.(13)
                PSMADE_position=x_powell;
                PSMADE_value=xp_fitness;
            end

            PSMADE_curve(it) = PSMADE_value;
            it = it + 2;
        end
    end
   %%
    
    a = atanh(-(it/Max_iteration)+1); 
    b = 1-it/Max_iteration; 
    % Update the Position of search agents
    for i=1:n
        if rand<z     %Eq.(3)
            x(i,:) = (ub-lb)*rand+lb;
        else
            p =tanh(abs(all_fitness(i)-PSMADE_value)); 
            vb = unifrnd(-a,a,1,dim);  
            vc = unifrnd(-b,b,1,dim);
            for j = 1:dim
                r = rand();
                A = randi([1,n]);  
                B = randi([1,n]);
                
                if r < p
                    x(i,j) = PSMADE_position(j) + vb(j) * (weight(i,j) * x(A,j) - x(B,j));
                else                   
                    x(i,j)=vc(j)*x(i,j);    
                end
            end
        end
    end
   
end
time = cputime - t;
end

function [x,pb,all_fitness,pb_fit] = funcDE(fobj,x,pb,all_fitness,pb_fit,lb,ub,beta_min,beta_max)
     %*****  Muation-Crossover  ****%
     n = size(x,1);
     dim=size(x,2);
     
     for i=1:1:n 
        tx=x(i,:); 
        
        %*****  Muation  ****%
        % Randomly select three individuals for mutation
        A=randperm(n); 
        A(A==i)=[];
        a=A(1);b=A(2);c=A(3);
        
        beta=unifrnd(beta_min,beta_max,dim); % Randomly generate a scaling factor
        v=zeros(1,dim);
        % Generate an intermediate individual
        for j=1:dim 
            v(j)=pb(a,j)+beta(j)*(pb(b,j)-pb(c,j)); %Eq.(4)
        end
        
        % Prevent intermediate individual from going out of bounds
        v=max(v,lb);
		v=min(v,ub);
        
        %*****  Crossover  ****%
        u=zeros(1,dim); 
        % Generate a pseudorandom number to select the dimension index for swapping
        j0=randi([1,dim]); 
        CR=0.5*(1+rand());
        for j=1:dim %Eq.(5)
            if j==j0 || rand<=CR 
                u(j)=v(j); 
            else
                u(j)=tx(j); 
            end
        end
       
        x_de=u;
        x_de_fit=fobj(x_de);   

        if x_de_fit<all_fitness(i) %Eq.(12)
           x(i,:)=x_de; 
           all_fitness(i)=x_de_fit;
        end
        if all_fitness(i)<pb_fit(i)
            pb(i,:)=x(i,:);
            pb_fit(i)=all_fitness(i);
        end
    end  
end