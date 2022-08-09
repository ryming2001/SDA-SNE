%% Source code is provided by 
%  Paper: 
%  Published on 
%  Contact: 

clear; 
close all;
clc;

%% Parameter Selection and Initialization
% DATASET = {'Easy/android'     ,'Easy/cube'   ,'Easy/fish'     ,Easy/goldfish  ,'Easy/polyhedron' ,'Easy/teeth'     ,'Easy/torus'      ,'Easy/torusknot',
%            'Medium/airballoon','Medium/bird' ,'Medium/bunny'  ,'Medium/chair' ,'Medium/fairy'    ,'Medium/penguin' ,'Medium/sealion'  ,'Medium/Trex',
%            'Hard/batman'      ,'Hard/knight' ,'Hard/pineapple','Hard/plant'   ,'Hard/priest'     ,'Hard/room'      ,'Hard/temple'     ,'Hard/tree'}
DATASET = 'test_data/Easy/torusknot';
SNE = 'SDA'; % SNE  = {'SDA': SNE introduced by us
             %         'OTW' : other SNEs}
BASE = 'CP2TV'; % BASE = {'CP2TV','3F2N'}
MAX_ITER = 3; % MAX_ITER = {3,inf} : maximum iterations
UMAX = 640; VMAX = 480;
PIC_NUM = 1;  % serial number of an image




%% Load Data
% K is the camera intrinsic matrix 
[K, ~] = get_dataset_params([DATASET,'/']);
[X, Y, Z, N, mask, u_map, v_map]... 
= load_data([DATASET,'/'], PIC_NUM, K, UMAX, VMAX);
% X, Y, Z are the x-, y-, and z- coordinates
% mask indicates the background and foreground
% UMAX --> image width, VMAX --> image height

nx_gt = N(:,:,1);
ny_gt = N(:,:,2);
nz_gt = N(:,:,3);
% ground-truth nx, ny, nz

ngt_vis = visualization_map_creation(nx_gt, ny_gt, nz_gt, mask);
% create a visualization map
%% SNE
if strcmp(SNE,'SDA')
    if strcmp(BASE,'CP2TV')
    % SDA with CP2TV
    [Gu,Gv]=SDA(Z,VMAX,UMAX,MAX_ITER);
    [nx_t,ny_t,nz_t]=CP2TV(Z,K,Gu,Gv,v_map,u_map);
    elseif strcmp(BASE,'3F2N')
    % SDA with 3F2N
    D = 1./Z; % inverse depth or disparity 
    [Gu,Gv]=SDA(D,VMAX,UMAX,MAX_ITER);
    [nx_t,ny_t,nz_t]=TF2N(X,Y,Z,K,Gu,Gv,VMAX,UMAX,'median');
    end
elseif strcmp(SNE,'OTW')
    if strcmp(BASE,'CP2TV')
    Gx_CP2N=[0,0,0;0,-2,2;0,0,0];
    Gy_CP2N=[0,0,0;0,-2,0;0,2,0];
    Gu = conv2(Z, Gx_CP2N, 'same')./(-2); 
    Gv = conv2(Z, Gy_CP2N, 'same')./(-2);
    [nx_t,ny_t,nz_t]=CP2TV(Z,K,Gu,Gv,v_map,u_map);
    elseif strcmp(BASE,'3F2N')
    kernel_size=3;
    [Gx_3F2N, Gy_3F2N] = set_kernel('fd', kernel_size); 
    D = 1./Z; % inverse depth or disparity 
    Gu = conv2(D, Gx_3F2N, 'same'); 
    Gv = conv2(D, Gy_3F2N, 'same');
    [nx_t,ny_t,nz_t]=TF2N(X,Y,Z,K,Gu,Gv,VMAX,UMAX,'median');
    end
end


[nx_t, ny_t, nz_t] = vector_normalization(nx_t, ny_t, nz_t);
% normalize the estimated surface normal
nt_vis = visualization_map_creation(nx_t, ny_t, nz_t, mask);
% create a visualization map
[error_map,error_vector, ea, ep] = ...
evaluation(nx_gt, ny_gt, nz_gt, nx_t, ny_t, nz_t, mask, VMAX, UMAX, 30);
% evaluation, ea and ep are explained in the paper
ep;
ea

%% Visualization
fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
ax1 = subplot(2,2,1); 
ngt_vis=rot90(ngt_vis,2);  % rotation
imshow(ngt_vis);
title(' Ground truth')
ax2 = subplot(2,2,2);
nt_vis=rot90(nt_vis,2);
imshow(nt_vis);
title('Result')
ax3 = subplot(2,2,[3,4]); 
error_map=rot90(error_map,2);
plot=imshow(error_map, [0,30],'Colormap',cool(4096)); 
set(plot,'alphadata',~isnan(error_map)) 
title('Error map (degrees)')
cur_colorb = colorbar;
cur_colorb.Label.String = 'error (degrees)';


%% additional functions
function [K, max_frm] = get_dataset_params(DATASET)
    fileID = fopen(['./', DATASET,'/params.txt'],'r');
    tmp = fscanf(fileID,'%f');
    fclose(fileID);
    fx = tmp(1); 
    fy = tmp(2);
    uo = tmp(3);
    vo = tmp(4);
    max_frm = tmp(5);
    K = [fx, 0, uo; 0, fy, vo; 0, 0, 1];
end

function [X, Y, Z, N, mask, u_map, v_map] = load_data(DATASET, frm, K, UMAX, VMAX)
    N = double(imread([DATASET, '/normal/',sprintf('%06d',frm),'.png']))/65535*2-1;
    N=-N;
    file_name = [DATASET, '/depth/',sprintf('%06d',frm),'.bin'];
    file_id = fopen(file_name);
    data = fread(file_id, 'float');
    fclose(file_id);

    u_map = ones(VMAX,1)*[1:UMAX] -K(1,3);
    v_map = [1:VMAX]'*ones(1,UMAX)-K(2,3);
    Z = reshape(data, UMAX, VMAX)';
    X = u_map.*Z/K(1,1);
    Y = v_map.*Z/K(2,2);
    
    % create mask
    mask = zeros(VMAX, UMAX);
    mask(Z==1) = 1; 

    
    nx_gt = N(:,:,1); ny_gt = N(:,:,2); nz_gt = N(:,:,3);
    %nx_gt = N(:,:,3); ny_gt = N(:,:,2); nz_gt = N(:,:,1);
    [nx_gt, ny_gt, nz_gt] = vector_normalization(nx_gt, ny_gt, nz_gt);
    N(:,:,1) = nx_gt; N(:,:,2) = ny_gt; N(:,:,3) = nz_gt;
end

function [nx,ny,nz]=vector_normalization(nx,ny,nz)
    mag=sqrt(nx.^2+ny.^2+nz.^2);
    nx=nx./mag;
    ny=ny./mag;
    nz=nz./mag;
end

function n_vis=visualization_map_creation(nx,ny,nz,mask)
% mask:1 is background, 0 is obj
% make background white,[1,1,1]
    [VMAX, UMAX] = size(nx);
    n_vis = zeros(VMAX,UMAX,3);
    n_vis(:,:,1) = nx.*(~mask)-mask;
    n_vis(:,:,2) = ny.*(~mask)-mask;
    n_vis(:,:,3) = nz.*(~mask)-mask;
    n_vis = (1-n_vis)/2; % [-1,1] transform to [0,1]
end


function angle_map = angle_normalization(angle_map)
    for i=1:numel(angle_map)
        if angle_map(i)>pi/2
            angle_map(i)=pi-angle_map(i);
        end
    end
end


function [error_map,error_vector, ea, ep] = evaluation(nx_gt, ny_gt, nz_gt, nx, ny, nz, mask, VMAX, UMAX, tr)
    scale = pi/180;
    error_map = acos(nx_gt.*nx+ny_gt.*ny+nz_gt.*nz);
    error_map = angle_normalization(error_map)./scale;
    error_map(mask==1) = nan;
    error_vector = reshape(error_map,[VMAX*UMAX,1]);
    error_vector(isnan(error_vector))=[];
    ea=mean(error_vector);
    ep = [];
    for j = tr:-5:5
    tmp = error_vector;
    tmp(error_vector > j) = [];
    ep_tmp = length(tmp) / length(error_vector);
    ep = [ep; ep_tmp];
    end
end

%% gradients estimator function
function [Gu,Gv] = SDA(Z,VMAX, UMAX,max_iter)
    %parameters

    % kernel setting
    u_laplace =[-1 2 -1];
    v_laplace =[-1;2;-1];

    Gx=[0,0,0;0.5,0,-0.5;0,0,0];
    GxL=[0,0,0;0,1,-1;0,0,0];
    GxR=[0,0,0;1,-1,0;0,0,0];
    Gy=[0,0.5,0;0,0,0;0,-0.5,0]; 
    GyU=[0,0,0;0,1,0;0,-1,0]; 
    GyD=[0,1,0;0,-1,0;0,0,0]; 

    % Initialization
    Z_ulaplace_0=conv2(Z,u_laplace,'same');
    Z_vlaplace_0=conv2(Z,v_laplace,'same');

    Z_ulaplace=abs(Z_ulaplace_0);
    Z_vlaplace=abs(Z_vlaplace_0);

    con_L=(1-sign([inf*ones(VMAX, 1), Z_ulaplace_0(:, 1:UMAX-1)].*[inf*ones(VMAX, 2), Z_ulaplace_0(:, 1:UMAX-2)]));
    con_R=(1-sign([Z_ulaplace_0(:, 2:UMAX), inf*ones(VMAX, 1)].*[Z_ulaplace_0(:, 3:UMAX), inf*ones(VMAX, 2)]));
    con_U=(1-sign([inf*ones(1, UMAX); Z_vlaplace_0(1:VMAX-1, :)].*[inf*ones(2, UMAX); Z_vlaplace_0(1:VMAX-2, :)]));
    con_D=(1-sign([Z_vlaplace_0(2:VMAX, :); inf*ones(1, UMAX)].*[Z_vlaplace_0(3:VMAX, :); inf*ones(2, UMAX)]));

    MAX=1e10;

    Gu_stack = cat(3,conv2(Z, Gx, 'same'),...
                        conv2(Z, GxR, 'same'),...
                        conv2(Z, GxL, 'same'));
    Gv_stack = cat(3,conv2(Z, Gy, 'same'),...
                        conv2(Z, GyD, 'same'),...
                        conv2(Z, GyU, 'same'));

    Eu_stack = cat(3,Z_ulaplace,...
                                [Z_ulaplace(:, 2:UMAX), inf*ones(VMAX, 1)],...
                                [inf*ones(VMAX, 1), Z_ulaplace(:, 1:UMAX-1)]);
    Ev_stack = cat(3,Z_vlaplace,...
                                [Z_vlaplace(2:VMAX, :); inf*ones(1, UMAX)],...
                                [inf*ones(1, UMAX); Z_vlaplace(1:VMAX-1, :)]);
    % state variable initialization
    [vmin_val,vmin_index] = min(Ev_stack,[],3);        
    [umin_val,umin_index] = min(Eu_stack,[],3);
    ulist_index=(umin_index-1)*VMAX*UMAX+reshape([1:VMAX*UMAX],VMAX,UMAX);
    vlist_index=(vmin_index-1)*VMAX*UMAX+reshape([1:VMAX*UMAX],VMAX,UMAX);
    % smoothness energy initialization
    Eu=Eu_stack(ulist_index);
    Ev=Ev_stack(vlist_index);
    % depth gradient initialization
    Gu = Gu_stack(ulist_index);
    Gv = Gv_stack(vlist_index);
    umin_index=4-umin_index;
    vmin_index=4-vmin_index;

    for i = 1 : max_iter
        % corresponding state variable:{[-1, 0],
        %                               [1, 0],
        %                               [0, 0],
        %                               [0, -1],
        %                               [0, 1],
        %                               [-1, -1],
        %                               [1, -1],
        %                               [-1, 1],
        %                               [1, 1]}
        temp_ulaplace_U=[inf*ones(1, UMAX); Z_ulaplace(1:VMAX-1, :)];
        temp_ulaplace_D=[Z_ulaplace(2:VMAX, :); inf*ones(1, UMAX)];
        temp_veval_U=[inf*ones(1, UMAX); Ev(1:VMAX-1, :)];
        temp_veval_D=[Ev(2:VMAX, :); inf*ones(1, UMAX)];
        Eu_stack=cat(3,[inf*ones(VMAX, 1), Z_ulaplace(:, 1:UMAX-1)]+(abs(umin_index-1)+abs([inf*ones(VMAX, 1), umin_index(:, 1:UMAX-1)]-1)+con_L)*MAX,...
                                [Z_ulaplace(:, 2:UMAX), inf*ones(VMAX, 1)]+(abs(umin_index-2)+abs([umin_index(:, 2:UMAX), inf*ones(VMAX, 1)]-2)+con_R)*MAX,...
                                Eu,...
                                2*([inf*ones(1, UMAX); Z_vlaplace(1:VMAX-1, :)]+[inf*ones(1, UMAX); Eu(1:VMAX-1, :)]),...
                                2*([Z_vlaplace(2:VMAX, :); inf*ones(1, UMAX)]+[Eu(2:VMAX, :); inf*ones(1, UMAX)]),...
                                [inf*ones(1, UMAX); Z_vlaplace(1:VMAX-1, :)]+[inf*ones(1, UMAX); Eu(1:VMAX-1, :)]+[inf*ones(VMAX, 1), temp_ulaplace_U(:, 1:UMAX-1)]+[inf*ones(VMAX, 1), temp_veval_U(:, 1:UMAX-1)],...
                                [inf*ones(1, UMAX); Z_vlaplace(1:VMAX-1, :)]+[inf*ones(1, UMAX); Eu(1:VMAX-1, :)]+[temp_ulaplace_U(:, 2:UMAX), inf*ones(VMAX, 1)]+[temp_veval_U(:, 2:UMAX), inf*ones(VMAX, 1)],...
                                [Z_vlaplace(2:VMAX, :); inf*ones(1, UMAX)]+[Eu(2:VMAX, :); inf*ones(1, UMAX)]+[inf*ones(VMAX, 1), temp_ulaplace_D(:, 1:UMAX-1)]+[inf*ones(VMAX, 1), temp_veval_D(:, 1:UMAX-1)],...
                                [Z_vlaplace(2:VMAX, :); inf*ones(1, UMAX)]+[Eu(2:VMAX, :); inf*ones(1, UMAX)]+[temp_ulaplace_D(:, 2:UMAX), inf*ones(VMAX, 1)]+[temp_veval_D(:, 2:UMAX), inf*ones(VMAX, 1)]);   

        % corresponding state variable:{[0, -1],
        %                               [0, 1],
        %                               [0, 0],
        %                               [-1, 0],
        %                               [1, 0],
        %                               [-1, -1],
        %                               [-1, 1],
        %                               [1, -1],
        %                               [1, 1]}
        temp_vlaplace_L=[inf*ones(VMAX, 1), Z_vlaplace(:, 1:UMAX-1)];
        temp_vlaplace_R=[Z_vlaplace(:, 2:UMAX), inf*ones(VMAX, 1)];
        temp_ueval_L=[inf*ones(VMAX, 1), Eu(:, 1:UMAX-1)];
        temp_ueval_R=[Eu(:, 2:UMAX), inf*ones(VMAX, 1)];
        Ev_stack=cat(3,[inf*ones(1, UMAX); Z_vlaplace(1:VMAX-1, :)]+(abs(vmin_index-1)+abs([inf*ones(1, UMAX); vmin_index(1:VMAX-1, :)]-1)+con_U)*MAX,...
                                [Z_vlaplace(2:VMAX, :); inf*ones(1, UMAX)]+(abs(vmin_index-2)+abs([vmin_index(2:VMAX, :); inf*ones(1, UMAX)]-2)+con_D)*MAX,...
                                Ev,...
                                2*([inf*ones(VMAX, 1), Z_ulaplace(:, 1:UMAX-1)]+[inf*ones(VMAX, 1), Ev(:, 1:UMAX-1)]),...
                                2*([Z_ulaplace(:, 2:UMAX), inf*ones(VMAX, 1)]+[Ev(:, 2:UMAX), inf*ones(VMAX, 1)]),...
                                [inf*ones(VMAX, 1), Z_ulaplace(:, 1:UMAX-1)]+[inf*ones(VMAX, 1), Ev(:, 1:UMAX-1)]+[inf*ones(1, UMAX); temp_vlaplace_L(1:VMAX-1, :)]+[inf*ones(1, UMAX); temp_ueval_L(1:VMAX-1, :)],...
                                [inf*ones(VMAX, 1), Z_ulaplace(:, 1:UMAX-1)]+[inf*ones(VMAX, 1), Ev(:, 1:UMAX-1)]+[temp_vlaplace_L(2:VMAX, :); inf*ones(1, UMAX)]+[temp_ueval_L(2:VMAX, :); inf*ones(1, UMAX)],...
                                [Z_ulaplace(:, 2:UMAX), inf*ones(VMAX, 1)]+[Ev(:, 2:UMAX), inf*ones(VMAX, 1)]+[inf*ones(1, UMAX); temp_vlaplace_R(1:VMAX-1, :)]+[inf*ones(1, UMAX); temp_ueval_R(1:VMAX-1, :)],...
                                [Z_ulaplace(:, 2:UMAX), inf*ones(VMAX, 1)]+[Ev(:, 2:UMAX), inf*ones(VMAX, 1)]+[temp_vlaplace_R(2:VMAX, :); inf*ones(1, UMAX)]+[temp_ueval_R(2:VMAX, :); inf*ones(1, UMAX)]);  


        temp_Gv_U=[zeros(1, UMAX); Gv(1:VMAX-1, :)];
        temp_Gv_D=[Gv(2:VMAX, :); zeros(1, UMAX)];
        Gu_stack=cat(3,Gu+1/(i+1)*Z-1/(i+1)*([zeros(VMAX, 1), Z(:, 1:UMAX-1)]+[zeros(VMAX, 1), Gu(:, 1:UMAX-1)]),...
                           Gu-1/(i+1)*Z+1/(i+1)*([Z(:, 2:UMAX), zeros(VMAX, 1)]-[Gu(:, 2:UMAX), zeros(VMAX, 1)]),...
                           Gu,...
                           [zeros(1, UMAX); Gu(1:VMAX-1, :)],...
                           [Gu(2:VMAX, :); zeros(1, UMAX)],...
                           [zeros(1, UMAX); Gu(1:VMAX-1, :)]-[zeros(VMAX, 1), temp_Gv_U(:, 1:UMAX-1)]+Gv,...
                           [zeros(1, UMAX); Gu(1:VMAX-1, :)]+[temp_Gv_U(:, 2:UMAX), zeros(VMAX, 1)]-Gv,...
                           [Gu(2:VMAX, :); zeros(1, UMAX)]+[zeros(VMAX, 1), temp_Gv_D(:, 1:UMAX-1)]-Gv,...
                           [Gu(2:VMAX, :); zeros(1, UMAX)]-[temp_Gv_D(:, 2:UMAX), zeros(VMAX, 1)]+Gv);
        temp_Gu_L=[zeros(VMAX, 1), Gu(:, 1:UMAX-1)];
        temp_Gu_R=[Gu(:, 2:UMAX), zeros(VMAX, 1)];
        Gv_stack=cat(3,Gv+1/(i+1)*Z-1/(i+1)*([zeros(1, UMAX); Z(1:VMAX-1, :)]+[zeros(1, UMAX); Gv(1:VMAX-1, :)]),...
                           Gv-1/(i+1)*Z+1/(i+1)*([Z(2:VMAX, :); zeros(1, UMAX)]-[Gv(2:VMAX, :); zeros(1, UMAX)]),...
                           Gv,...
                           [zeros(VMAX, 1), Gv(:, 1:UMAX-1)],...
                           [Gv(:, 2:UMAX), zeros(VMAX, 1)],...
                           [zeros(VMAX, 1), Gv(:, 1:UMAX-1)]-[zeros(1, UMAX); temp_Gu_L(1:VMAX-1, :)]+Gu,...
                           [zeros(VMAX, 1), Gv(:, 1:UMAX-1)]+[temp_Gu_L(2:VMAX, :); zeros(1, UMAX)]-Gu,...
                           [Gv(:, 2:UMAX), zeros(VMAX, 1)]+[zeros(1, UMAX); temp_Gu_R(1:VMAX-1, :)]-Gu,...
                           [Gv(:, 2:UMAX), zeros(VMAX, 1)]-[temp_Gu_R(2:VMAX, :); zeros(1, UMAX)]+Gu);
        % state trandfer
        [umin_val,umin_index] = min(Eu_stack,[],3);
        [vmin_val,vmin_index] = min(Ev_stack,[],3);
        if and(all(umin_index(:)==3),all(vmin_index(:)==3))   % if all state variables reach stable 
            i
            break;
        end
        ulist_index=(umin_index-1)*VMAX*UMAX+reshape([1:VMAX*UMAX],VMAX,UMAX);
        vlist_index=(vmin_index-1)*VMAX*UMAX+reshape([1:VMAX*UMAX],VMAX,UMAX);
        % smoothness energy minimization
        Eu=Eu_stack(ulist_index);
        Ev=Ev_stack(vlist_index);
        % depth gradient update
        Gu = Gu_stack(ulist_index);  
        Gv = Gv_stack(vlist_index);  
    end
end


%% normal estimator function

% CP2TV
function [nx_t,ny_t,nz_t]=CP2TV(Z,K,Gu,Gv,v_map,u_map)
    nx_t = Gu*K(1,1);
    ny_t = Gv*K(2,2);
    nz_t = -(Z + v_map.*Gv + u_map.*Gu);
end

% 3F2N
function [Gx, Gy] = set_kernel(kernel_name, kernel_size)
    if strcmp(kernel_name, 'fd')
        Gx = zeros(kernel_size);
        mid = (kernel_size + 1) / 2;
        temp = 1;
        for index = (mid+1) : kernel_size
            Gx(mid, index) = temp;
            Gx(mid, 2*mid-index) = -temp;
            temp = temp + 1;
        end
        Gy = Gx';
    elseif strcmp(kernel_name, 'sobel') | strcmp(kernel_name, 'scharr') | strcmp(kernel_name, 'prewitt')
        if strcmp(kernel_name, 'sobel')
            smooth = [1 2 1];
        elseif strcmp(kernel_name, 'scharr')
            smooth = [1 1 1];
        else
            smooth = [3 10 3];
        end
        kernel3x3 = smooth' * [-1 0 1];
        kernel5x5 = conv2(smooth' * smooth, kernel3x3);
        kernel7x7 = conv2(smooth' * smooth, kernel5x5);
        kernel9x9 = conv2(smooth' * smooth, kernel7x7);
        kernel11x11 = conv2(smooth' * smooth, kernel9x9);
        if kernel_size == 3
            kernel = kernel3x3; Gx = kernel; Gy = Gx';
        elseif kernel_size == 5
            kernel = kernel5x5; Gx = kernel; Gy = Gx';
        elseif kernel_size == 7
            kernel = kernel7x7; Gx = kernel; Gy = Gx';
        elseif kernel_size == 9
            kernel = kernel9x9; Gx = kernel; Gy = Gx';
        elseif kernel_size == 11
            kernel = kernel11x11; Gx = kernel; Gy = Gx';
        end
    end
end
function [X_d, Y_d, Z_d] = delta_xyz_computation(X,Y,Z,pos)
    if pos==1
        kernel=[0,-1,0;0,1,0;0,0,0];
    elseif pos==2
        kernel=[0,0,0;-1,1,0;0,0,0];
    elseif pos==3
        kernel=[0,0,0;0,1,-1;0,0,0];
    elseif pos==4
        kernel=[0,0,0;0,1,0;0,-1,0];
    elseif pos==5
        kernel=[-1,0,0;0,1,0;0,0,0];
    elseif pos==6
        kernel=[0,0,0;0,1,0;-1,0,0];
    elseif pos==7
        kernel=[0,0,-1;0,1,0;0,0,0];
    else
        kernel=[0,0,0;0,1,0;0,0,-1];
    end

    X_d = conv2(X, kernel, 'same');
    Y_d = conv2(Y, kernel, 'same');
    Z_d = conv2(Z, kernel, 'same');

    X_d(Z_d==0) = nan;
    Y_d(Z_d==0) = nan;
    Z_d(Z_d==0) = nan;
end
function [nx_t,ny_t,nz_t]=TF2N(X,Y,Z,K,Gu,Gv,VMAX,UMAX,third_filter)
    nx_t = Gu*K(1,1);
    ny_t = Gv*K(2,2);
    % estimated nx and ny
    nz_t_volume = zeros(VMAX, UMAX, 8);
    % create a volume to compute nz                

    for j = 1:8
        [X_d, Y_d, Z_d] = delta_xyz_computation(X, Y, Z, j);
        nz_j = -(nx_t.*X_d+ny_t.*Y_d)./Z_d; % the background Z_d is 0£¬nz_j is nan
        nz_t_volume(:,:,j) = nz_j;
    end

    if strcmp(third_filter, 'median')
        nz_t = nanmedian(nz_t_volume, 3);
    else
        nz_t = nanmean(nz_t_volume, 3);
    end

    nx_t(isnan(nz_t))=0;
    ny_t(isnan(nz_t))=0;
    nz_t(isnan(nz_t))=-1;
    % process infinite points
end