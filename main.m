%% PSMADE
clear all
clc
tic;
addpath(genpath(pwd));
n=30;   % Population size
Max_iteration=500;  % Maximum iteration count
Function_name='F6'; % Fitness Function F1-F23
        
% lb = Lower boundary
% ub: Upper boundary
% dim: The dimensionality of the problem
% fobj: objective function	
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); % Get test function information
    
[Best_pos,Best_score,PSMADE_curve]=PSMADE(n,Max_iteration,lb,ub,dim,fobj);

figure('Position',[269   240   660   290])
% Draw search space
subplot(1,2,1);
func_plot(Function_name);
title('Parameter space')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])

% Draw objective space
subplot(1,2,2);
semilogx(PSMADE_curve,'Color','r','linewidth',1.5)
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');

axis tight
grid on
box on
legend('PSMADE')

display(['The best solution obtained by ,PSMADE is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by ,PSMADE is : ', num2str(Best_score)]);

        
