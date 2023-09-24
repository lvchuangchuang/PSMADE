% function [Destination_fitness,bestPositions,time,Convergence_curve]=isma1(N,Max_FEs,lb,ub,dim,fobj)
function [Destination_position,Convergence_curve]=PSMADE(n,Max_FEs,lb,ub,dim,fobj)
% powell+DE
disp('PSMADE is now tackling your problem')
t = cputime;
% initialize position
Destination_position=zeros(1,dim);
Destination_fitness=inf;%change this to -inf for maximization problems
all_fitness = inf*ones(n,1);%record the fitness of all slime mold
weight = ones(n,dim);%fitness weight of each slime mold
%Initialize the set of random solutions
x=initialization(n,dim,ub,lb);
Max_iteration=Max_FEs;  %Maximum number of evaluations
Convergence_curve=[];
fes=0;   %Number of current evaluations
% MaxFEs=Max_FEs;  %Maximum number of evaluations
it = 1;
lb=ones(1,dim).*lb; % lower boundary
ub=ones(1,dim).*ub; % upper boundary
z=0.03; % parameter0.03
beta_min=0.4; % Lower Bound of Scaling Factor
beta_max=1; % Upper Bound of Scaling Factor
pb=zeros(n,dim);  % Record the optimal solution for each individual
pb_fit=inf*ones(n,1);
taboo=[];

% Main loop
while  fes < Max_iteration
    %sort the fitness
    for i=1:n
        % Check if solutions go outside the search space and bring them back
        Flag4ub=x(i,:)>ub;
        Flag4lb=x(i,:)<lb;
        x(i,:)=(x(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        all_fitness(i) = fobj(x(i,:));
        fes = fes+1;
        if all_fitness(i) < Destination_fitness 
            Destination_fitness = all_fitness(i);
            Destination_position = x(i,:);
        end
        if all_fitness(i)<pb_fit(i)
            pb(i,:)=x(i,:);
            pb_fit(i)=all_fitness(i);
        end
    end
    
     
    Convergence_curve(it) = Destination_fitness;
    it = it + 1;
    
    [x,pb,all_fitness,pb_fit,tmp_fes] = funcDE(fobj,x,pb,all_fitness,pb_fit,lb,ub,beta_min,beta_max);
    fes = fes+tmp_fes;
    
    [smell_order,smell_index] = sort(all_fitness);  %Eq.(2.6)
    worst_fitness = smell_order(n);
    best_fitness = smell_order(1);

    S=best_fitness-worst_fitness+eps;  % plus eps to avoid denominator zero
    
     %update the best fitness value and best position
    if best_fitness < Destination_fitness
        Destination_position=x(smell_index(1),:);
        Destination_fitness = best_fitness;
    end

    %calculate the fitness weight of each slime mold
    for i=1:n
        for j=1:dim
            if i<=(n/2)    %Eq.(2.5)
                weight(smell_index(i),j) = 1+rand()*log10((best_fitness-smell_order(i))/(S)+1);
            else
                weight(smell_index(i),j) = 1-rand()*log10((best_fitness-smell_order(i))/(S)+1);
            end
        end
    end
    
    
    if fes>0.8*Max_FEs
        %% Powell
         % Determine if it is in the Tabu List
        [M,~] = size(taboo);  
        flag = 0;
        for m = 1:M
            if Destination_position == taboo(m,1) & Destination_fitness == taboo(m,2)
                flag = 1;
            end
        end
        if flag==0
            % Update the Tabu List
            taboo = [taboo;[Destination_position,Destination_fitness]];
            [xo,Ot,nS]=powell(fobj,Destination_position',0,1,lb,ub,-1,1e-6);
            xo = xo';
            if Ot<Destination_fitness
                Destination_position=xo;
                Destination_fitness=Ot;
            end
            fes=fes+nS;
             Convergence_curve(it:it+nS-1) = Destination_fitness;
             it = it + nS;
        end
    end
   %%
    
    a = atanh(-(fes/Max_FEs)+1); %Eq.(2.4)
    b = 1-fes/Max_FEs; 
    % Update the Position of search agents
    for i=1:n
        if rand<z     %Eq.(2.7)
            x(i,:) = (ub-lb)*rand+lb;
        else
            p =tanh(abs(all_fitness(i)-Destination_fitness)); %Eq.(2.2)
            vb = unifrnd(-a,a,1,dim);  %Eq.(2.3)
            vc = unifrnd(-b,b,1,dim);
            for j = 1:dim
                r = rand();
                A = randi([1,n]);  
                B = randi([1,n]);
                
                if r < p
                    x(i,j) = Destination_position(j) + vb(j) * (weight(i,j) * x(A,j) - x(B,j));
                else                   
                    x(i,j)=vc(j)*x(i,j);    
                end
            end
        end
    end
   
end
time = cputime - t;
end

function [x,pb,all_fitness,pb_fit,tmp_fes] = funcDE(fobj,x,pb,all_fitness,pb_fit,lb,ub,beta_min,beta_max)
     %*****Muation-Crossover****%%%
     n = size(x,1);
     dim=size(x,2);
     tmp_fes=0;
     for i=1:1:n % Iterate through each individual
        tx=x(i,:); % Extract individual locations
  
        % Three individuals are randomly selected for use in case of variation
        A=randperm(n); % The order of the individuals is re-randomized
        A(A==i)=[]; % The current individual is vacated (the current individual is not involved when the mutated intermediate is generated)
        a=A(1);b=A(2);c=A(3);
        % Mutation   
%         T=4*Max_FEs;w=2*pi/T;beta=cos(w*fes)*(beta_max-beta_min)+beta_min;
        beta=unifrnd(beta_min,beta_max,dim); % Randomly generate scaling factors
        y=zeros(1,dim);
        for j=1:dim
            y(j)=pb(a,j)+beta(j)*(pb(b,j)-pb(c,j)); % Produce intermediates
        end
        
        % Prevent intermediates from crossing boundaries
        y=max(y,lb);
		y=min(y,ub);
        % 交叉操作 Crossover
        tz=zeros(1,dim); % Initialize a new individual
        j0=randi([1,dim]); % Generate a pseudo-random number, that is, select the dimension number to be exchanged
        CR=0.5*(1+rand());
        for j=1:dim % Traverse each dimension
            if j==j0 || rand<=CR % If the current dimension is the dimension to be exchanged or the random probability is less than the crossover probability
                tz(j)=y(j); % The current dimension value of the new individual is equal to the corresponding dimension value of the intermediate body
            else
                tz(j)=tx(j); % The current dimension value of the new individual is equal to the corresponding dimension value of the current individual
            end
        end
       
        tz_fit=fobj(tz); % New individual objective function value
        tmp_fes=tmp_fes+1;
        if tz_fit<all_fitness(i) % If the new individual is better than the current individual
           x(i,:)=tz; % Update current individual
           all_fitness(i)=tz_fit;
        end
        if all_fitness(i)<pb_fit(i)
            pb(i,:)=x(i,:);
            pb_fit(i)=all_fitness(i);
        end
    end  
    %%
end
