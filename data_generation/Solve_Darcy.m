
clc; clear all; close all;
% this code solves the darcy flow problem on a square domain
% the code solves for u_x1,u_x2, and p of the following problem:
%
% div(u(x1,x2)) = f(x1,x2)    u = [u_x1;u_x2]
% u(x1,x2) = -K(x1,x2) grad(p(x1,x2))
% dot(u,n) = 0 on boundary
% integral(boundary) p dx1 dx2 = 0

% (see https://arxiv.org/pdf/1801.06879.pdf for problem details)


nx1 = 65; nx2 = 65;
kle = 3;

xv1 = linspace(0,1,nx1); xv2 = linspace(0,1,nx2);

corr.name = 'gauss1';
corr.c0 = 0.1; 

dx1=1/(nx1-1);
dx2=1/(nx2-1);

Areg = enforce_int(xv1,xv2,dx1,dx2); % enforces the integral condition

[X,Y] = meshgrid(xv1,xv2); mesh = [X(:) Y(:)]; % 2-D mesh

%{
var = [.01, 0;0, .01]; offset = .75; scale = 0.25;
center = [0.42, 0.59]; %k1 = 1; k2 = 2;r = .1;
K = gauss_K(X, Y, center, var, offset, scale);

n_data = 1024;
Zmat = zeros(kle,n_data);
Kmat = zeros(nx1,nx2,1,n_data);
Output = zeros(nx1,nx2,3,n_data);
for i = 1:n_data
    strcat(num2str(i/n_data*100),'%')
    c = 0.05*randn(1, 2) + [0.5, 0.5];
    K = gauss_K(X, Y, c, var, offset, scale);
    [A,f] = form_matrix(xv1,xv2,K,dx1,dx2);
    A = [A;Areg]; f = [f;0];
    P = A\f;
    P = reshape(P,nx1,nx2);
    [U1,U2] = compute_u(P,K,nx1,nx2,xv1,xv2);
    Zmat(:,i) = c';
    Kmat(:,:,1,i) = flipud(K');
    Output(:,:,1,i) = flipud(P');
    Output(:,:,2,i) = flipud(U1');
    Output(:,:,3,i) = flipud(U2');
    
end
%}
%
corr.sigma = 1.0;    % variance (sill)
%

[K,KL,U,L] = randomfield(corr,mesh,'trunc', kle);   % This is where we change the number of KL modes
figure(100);
plot(diag(L^2));



% want to generate data from a square domain on the parameters (2 only)
%{
nz1 = 32; nz2 = 32;
z1v = linspace(-3,3,nz1);
z2v = linspace(-3,3,nz2);
%}
n_data = 1024
Zmat = zeros(kle,n_data);
Kmat = zeros(nx1,nx2,1,n_data);
Output = zeros(nx1,nx2,3,n_data);

ind = 0;
for i = 1:n_data%nz1
    %for j = 1:nz2
        ind = ind + 1
        %W = randn(kle, 1)*.5 + 1;
        W = zeros(kle, 1);
        for j = 1:kle
            uni1 = rand(1);
            if uni1<= 0.5
                W(j) = randn(1)*0.25 + 1;%[z1v(i);z2v(j)];
            else
                W(j) = randn(1)*0.25 - 1;
            end
        end
        
        K = (U*L)*W;
        K = reshape(exp(K),nx1,nx2);
        [A,f] = form_matrix(xv1,xv2,K,dx1,dx2);
        A = [A;Areg]; f = [f;0];
        P = A\f;
        P = reshape(P,nx1,nx2);
        [U1,U2] = compute_u(P,K,nx1,nx2,xv1,xv2);
        Zmat(:,ind) = W';
        Kmat(:,:,1,ind) = flipud(K');
        Output(:,:,1,ind) = flipud(P');
        Output(:,:,2,ind) = flipud(U1');
        Output(:,:,3,ind) = flipud(U2');
    %end
end
inds = randperm(n_data);
Zmat = Zmat(:,inds); Kmat = Kmat(:,:,:,inds); Output = Output(:,:,:,inds);

Z2 = zeros(size(Zmat)); K2 = zeros(size(Kmat)); O2 = zeros(size(Output));
%}


create_hdf5(kle,32,Zmat,Kmat,Output);
create_hdf5(kle,64,Zmat,Kmat,Output);
create_hdf5(kle,128,Zmat,Kmat,Output);
create_hdf5(kle,256,Zmat,Kmat,Output);
create_hdf5(kle,512,Zmat,Kmat,Output);
create_hdf5_test_data(kle,512,Zmat,Kmat,Output);


%%% ------------------------------------------------------------------
%{
K2  =reshape(K, nx1, nx2);
K = (U*L)*randn(kle,1);
K=reshape(exp(K), nx1, nx2);


%K = ones(nx1,nx2);
%}
figure(5);
contourf(X, Y, K);colorbar;
[A,f] = form_matrix(xv1,xv2,K,dx1,dx2);

A = [A;Areg]; f = [f;0];
[X,Y] = meshgrid(xv1,xv2);
P = A\f;
P = reshape(P,nx1,nx2);
figure(1);hold on;
contourf(X,Y,P); colorbar;

[U1,U2] = compute_u(P, K, nx1, nx2, xv1, xv2);
figure(2); hold on; 
contourf(X,Y,U1);colorbar;
figure(3); hold on;
contourf(X,Y,U2);colorbar;
figure(1);
%{
figure(10); 
subplot(2,2,1);
imshow(flipud(K), [min(min(K)), max(max(K))]); colormap(jet(256));colorbar; title('Permeability $K$','Interpreter','latex');
subplot(2,2,2);
imshow(flipud(P), [min(min(P)), max(max(P))]); colormap(jet(256));colorbar; title('Pressure $p$','Interpreter','latex');
subplot(2,2,3);
imshow(flipud(U1), [min(min(U1)), max(max(U1))]); colormap(jet(256));colorbar; title('Velocity $u_y$','Interpreter','latex');
subplot(2,2,4);
imshow(flipud(U2), [min(min(U2)), max(max(U2))]); colormap(jet(256));colorbar; title('Velocity $u_x$','Interpreter','latex');
saveas(gcf, strcat('KLE',num2str(kle),'_data.png'));
%}
function [A] = enforce_int(xv1,xv2,dx1,dx2)
    nx1 = length(xv1);
    nx2 = length(xv2);
    A = zeros(1,nx2*nx1);
    j = 0;
    i = 0;
    for ii = 1:nx1*nx2
        if rem(ii-1,nx1)==0 
            j = j+1;
            i = 1;
        end
        
        if (i==1&j==1) %bottom left corner
            A(ii) = 1;
        elseif (i == nx1&j == nx2) % top right corner
            A(ii) = 1; 
        elseif (i==1&j==nx2) % top left corner
            A(ii) = 1;
        elseif (i==nx1&j==1) % bottom right corner
            A(ii) = 1; 
        elseif i == 1 % left boundary w/out corners
            A(ii) = 2;
        elseif j == 1 %bottom boundary w/out corners
            A(ii) = 2;
        elseif i == nx1 %right boundary w/out corners
            A(ii) = 2;
        elseif j == nx2
            A(ii) = 2;
        else
            A(ii) = 4;
        end
    end
    A = A * dx1 * dx2 / 4;
end


function create_hdf5(kle,n,Z,K,O)
    filename = strcat('kle',num2str(kle),'_lhs',num2str(n),'_bimodal_2.hdf5');
    %filename = strcat('gauss_location_only_big_condensed_lhs',num2str(n),'_gaussian.hdf5');
    h5create(filename,'/generative_params',[kle,n],'Datatype','single');
    h5create(filename,'/permeability',[size(K,1),size(K,2),1,n],'Datatype','single');
    h5create(filename,'/output',[size(O,1),size(O,2),3,n],'Datatype','single');
    
    h5write(filename,'/generative_params',Z(:,1:n));
    h5write(filename,'/permeability',K(:,:,:,1:n));
    h5write(filename,'/output',O(:,:,:,1:n));
    
    h5disp(filename);
end

function create_hdf5_test_data(kle,n,Z,K,O)
    filename = strcat('kle',num2str(kle),'_mc',num2str(n),'_bimodal_2.hdf5');
    %filename = strcat('gauss_location_only_big_condensed_mc',num2str(n),'_gaussian.hdf5');
    h5create(filename,'/generative_params',[kle,n],'Datatype','single');
    h5create(filename,'/permeability',[size(K,1),size(K,2),1,n],'Datatype','single');
    h5create(filename,'/output',[size(O,1),size(O,2),3,n],'Datatype','single');
    
    h5write(filename,'/generative_params',Z(:,n+1:end));
    h5write(filename,'/permeability',K(:,:,:,n+1:end));
    h5write(filename,'/output',O(:,:,:,n+1:end));
    
    h5disp(filename);
end

function [U1, U2] = compute_u(P,K,nx1,nx2,xv1,xv2)
    U1 = zeros(size(P)); U2 = zeros(size(P));
    dx1 = xv1(2)-xv1(1); dx2 = xv2(2)-xv2(1);
    for i = 1:length(xv1)
        for j = 1:length(xv2)
            if (j == 1 | j == nx2) & ~(i==1) & ~(i==nx1) % on bottom boundary or top boundary
                U2(i,j) = 0;
                U1(i,j) = -K(i,j)*(P(i+1,j)-P(i-1,j))/(2*dx1);
            elseif (i == 1 | i == nx1) & ~(j==1) & ~(j==nx2) % on left or right boundary
                U1(i,j) = 0;
                U2(i,j) = -K(i,j)*(P(i,j+1)-P(i,j-1))/(2*dx2);
            elseif (i==1&j==1)|(i == nx1&j == nx2)|(i==1&j==nx2)|(i==nx1&j==1) % on corners
                U1(i,j) = 0;
                U2(i,j) = 0;
            else
                U1(i,j) = -K(i,j)*(P(i+1,j)-P(i-1,j))/(2*dx1);
                U2(i,j) = -K(i,j)*(P(i,j+1)-P(i,j-1))/(2*dx2);

            end
        end
    end
end

function [dK] = grad_K(K,i,j,dx1,dx2,dim)
    % computes the gradient of K w.r.t dimension dim (1 = x1, 2 = x2) at
    % location x1(i), x2(j)
    dK = 0; % no error catch
    if dim == 1
        dK = (K(i+1,j)-K(i-1,j))/(2*dx1);
    elseif dim == 2
        dK = (K(i,j+1)-K(i,j-1))/(2*dx2);
    end

end

function [A,f] = form_matrix(xv1, xv2, K, dx1, dx2)
    nx1 = length(xv1); nx2 = length(xv2);
    A = zeros(nx1*nx2,nx2*nx1);
    f = zeros(nx1*nx2,1);
    % interior points
    j = 0;
    i = 0;
    for ii = 1:nx1*nx2
        
        if rem(ii-1,nx1)==0 
            j = j+1;
            i = 1;
        end
        
        if (i==1&j==1) %bottom left corner
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii+1) = -2*K(i,j)/dx1^2;
            A(ii,ii+nx1) = -2*K(i,j)/dx2^2;
        elseif (i == nx1&j == nx2) % top right corner
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii-1) = -2*K(i,j)/dx1^2;
            A(ii,ii-nx1) = -2*K(i,j)/dx2^2; 
        elseif (i==1&j==nx2) % top left corner
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii+1) = -2*K(i,j)/dx1^2;        
            A(ii,ii-nx1) = -2*K(i,j)/dx2^2; 
        elseif (i==nx1&j==1) % bottom right corner
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii-1) = -2*K(i,j)/dx1^2;
            A(ii,ii+nx1) = -2*K(i,j)/dx2^2; 
        elseif i == 1 % left boundary w/out corners
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii+1) = -2*K(i,j)/dx1^2;
            A(ii,ii+nx1) = -K(i,j)/dx2^2 - grad_K(K,i,j,dx1,dx2,2)/(2*dx2); % - add derivative of K w.r.t x2 / 2dx2
            A(ii,ii-nx1) = -K(i,j)/dx2^2 + grad_K(K,i,j,dx1,dx2,2)/(2*dx2); % + add derivative of K w.r.t x2 / 2dx2
        elseif j == 1 %bottom boundary w/out corners
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii-1) = -K(i,j)/dx1^2 + grad_K(K,i,j,dx1,dx2,1)/(2*dx1); % + add derivative of K w.r.t x1 / 2dx1
            A(ii,ii+1) = -K(i,j)/dx1^2 - grad_K(K,i,j,dx1,dx2,1)/(2*dx1); % - add derivative of K w.r.t x1 / 2dx1
            A(ii,ii+nx1) = -2*K(i,j)/dx2^2;
        elseif i == nx1 %right boundary w/out corners
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii-1) = -2*K(i,j)/dx1^2;
            A(ii,ii+nx1) = -K(i,j)/dx2^2 - grad_K(K,i,j,dx1,dx2,2)/(2*dx2); % - add derivative of K w.r.t x2 / 2dx2
            A(ii,ii-nx1) = -K(i,j)/dx2^2 + grad_K(K,i,j,dx1,dx2,2)/(2*dx2); % + add derivative of K w.r.t x2 / 2dx2
        elseif j == nx2
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii-1) = -K(i,j)/dx1^2 + grad_K(K,i,j,dx1,dx2,1)/(2*dx1); % + add derivative of K w.r.t x1 / 2dx1
            A(ii,ii+1) = -K(i,j)/dx1^2 - grad_K(K,i,j,dx1,dx2,1)/(2*dx1); % - add derivative of K w.r.t x1 / 2dx1
            A(ii,ii-nx1) = -2*K(i,j)/dx2^2;
        else
            A(ii,ii) = K(i,j)*2*(1/dx1^2+1/dx2^2);
            A(ii,ii-1) = -K(i,j)/dx1^2 + grad_K(K,i,j,dx1,dx2,1)/(2*dx1); % + add derivative of K w.r.t x1 / 2dx1
            A(ii,ii+1) = -K(i,j)/dx1^2 - grad_K(K,i,j,dx1,dx2,1)/(2*dx1); % - add derivative of K w.r.t x1 / 2dx1
            A(ii,ii+nx1) = -K(i,j)/dx2^2 - grad_K(K,i,j,dx1,dx2,2)/(2*dx2); % - add derivative of K w.r.t x2 / 2dx2
            A(ii,ii-nx1) = -K(i,j)/dx2^2 + grad_K(K,i,j,dx1,dx2,2)/(2*dx2); % + add derivative of K w.r.t x2 / 2dx2
            
        end
        x = xv1(i); y = xv1(j);
        
        %source function
        if abs(x-0.0625)<=0.0625 & abs(y-0.0625)<=0.0625
            f(ii) = 10;%20*cos(3*pi*x)*sin(2*pi*y);
        elseif abs(x-1+0.0625)<=0.0625 & abs(y-1+0.0625)<=0.0625
            f(ii) = -10;
        end
        i = i+1;
    end
    



end




