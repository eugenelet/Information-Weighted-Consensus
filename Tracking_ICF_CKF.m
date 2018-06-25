%%%%%%%%%%%%%%%%%%%%%%%
% Fixed Parameters    %
%%%%%%%%%%%%%%%%%%%%%%%
clear;
set(gcf, 'Position', get(0, 'Screensize'));
phi = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];  % State Transition Matrix
Q = [10 0 0 0; 0 10 0 0; 0 0 1 0; 0 0 0 1];  % Process Covariance
H = [1 0 0 0; 0 1 0 0];  % Observation Matrix
A = [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0];  % Communication Adjacency Matrix
B = 0.02 * eye(2);  % Measurement Information
nu = inv(B);  % ground_truth. Noise Covariance
deg = 2;  % Network degree
K = 2;  % Total Consensus Iteration
eps = 0.65 / deg;  % Consensus rate parameter
sensor_n = 5;  % Number of sensors
time_steps = 140;  % Number of time steps for target
% Field of View
FOV = cat(3, [13 13; 357 12; 200 350; 13 13], [100 130; 400 130; 300 490; 100 130], [250 490; 490 492; 400 200; 250 490], [10 490; 12 170; 270 300; 10 490], [200 150; 490 14; 440 290; 200 150]);
% hold on;
% for i=1:5
%   plot(FOV(:,1,i),FOV(:,2,i))
%   xlim([0 500])
%   ylim([0 500])
% end
% hold off;
%%%%%%%%%%%%%%%%%%%%%%%
% Initital Parameters %
%%%%%%%%%%%%%%%%%%%%%%%
% Initial location
x = [250;250;0;0];
% Set initial velocity
angle = 2*pi*rand(1);
speed = randi([10 20]);
x(3) = speed * cos(angle);
x(4) = speed * sin(angle);


P = [100 0 0 0; 0 100 0 0; 0 0 10 0; 0 0 0 10];  % Prior Covariance
W = inv(P);  % Prior information matrix
x_ICF = x + mvnrnd(zeros(4,1), P)';  % Prior estimate

node = [];  % nodes for ICF
for i = 1:5
  node = [node, ICF_node(x_ICF, W, H, eps, K, phi, Q)];
end
% Centralized Kalman Filter
CKF = CKF_(x_ICF, W, H, phi, Q);

%%%%%%%%%%%%%%%%%%%%%%%
% Start Tracking!!    %
%%%%%%%%%%%%%%%%%%%%%%%
sct = [];
sct_pred = [];
CKF_sct_pred = [];
obs = [];
target_loc = [];
pred_loc = {};
CKF_pred_loc = [];
pred_cov = {};
for i = 1:time_steps
    % Update state of target
    x = phi*x + mvnrnd(zeros(4,1), Q)';
    % Ground truth track of target
    ground_truth = H*x;
    % Observed track of target
    obs = [obs, H*x + mvnrnd(zeros(2,1), nu)'];
    % GT track for visibility purpose
    target_loc = [target_loc, ground_truth];

    % Invert velocity at boundaries
    if(x(1)<=0)
      x(3) = abs(x(3));
    end
    if(x(1)>=500)
      x(3) = -abs(x(3));
    end
    if(x(2)<=0)
      x(4) = abs(x(4));
    end
    if (x(2)>=500)
      x(4) = -abs(x(4));
    end

    % CKF
    CKF = CKF.compute_information(B, obs(:,end));
    [CKF, CKF_est_W, CKF_est_x] = CKF.compute_posterior_state_and_info();
    CKF = CKF.predict_next_step();
    CKF_est_z = H*CKF_est_x;

    if i == 1
      CKF_pred_loc = CKF_est_z;
    else
      CKF_pred_loc = [CKF_pred_loc, CKF_est_z];
    end

    % Compute information matrix and vector
    for j = 1:sensor_n
      [obs_in, obs_on] = inpolygon(obs(1,end),obs(2,end),FOV(:,1,j),FOV(:,2,j));
      obs_in_on = obs_in | obs_on;
      if obs_in_on  % observed by sensor
        node(j) = node(j).compute_information(B, obs(:,end));
      else
        node(j) = node(j).compute_information(0*B, obs(:,end));
      end
    end


    % Perform average consensus
    for k = 1:K
      for j = 1:sensor_n
        node(j).adj_V = {};
        node(j).adj_v = {};
        for adj = 1:sensor_n
          if A(j,adj)
            node(j).adj_V{size(node(j).adj_V, 2) + 1} = node(adj).V;
            node(j).adj_v{size(node(j).adj_v, 2) + 1} = node(adj).v;
          end
        end
      end
      for j = 1:sensor_n
        node(j) = node(j).average_consensus();
      end
    end

    % Plot 5 Field of View (FOV)
    if ~isempty(sct)
      for m=1:length(sct)
        delete(sct(m));
        delete(sct_pred(m));
        delete(CKF_sct_pred(m));
      end
    end
    for j = 1:sensor_n
      ax = subplot(3,2,j);
      hold on;

      % Compute posterior state estimate and information matrix for time t
      [node(j), est_W, est_x] = node(j).compute_posterior_state_and_info();
      % Prediction
      node(j) = node(j).predict_next_step();
      est_z = H*est_x;

      if i == 1
        pred_loc{j} = est_z;
        cov_tmp = inv(est_W);
        pred_cov{j} = cov_tmp(1);
      else
        pred_loc{j} = [pred_loc{j}, est_z];
        cov_tmp = inv(est_W);
        pred_cov{j} = [pred_cov{j}, cov_tmp(1)];
      end
      scatter(ax, pred_loc{j}(1,:),pred_loc{j}(2,:), pred_cov{j}, 'MarkerEdgeColor', 'k', ...
              'MarkerFaceColor', [220 220 220]/255, 'MarkerFaceAlpha',.8);
      sct_pred(j) = scatter(ax, est_z(1),est_z(2), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
      line(pred_loc{j}(1,:), pred_loc{j}(2,:), 'color', 'r')

      % CKF (For comparison)
      CKF_sct_pred(j) = scatter(ax, CKF_est_z(1),CKF_est_z(2), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
      line(CKF_pred_loc(1,:), CKF_pred_loc(2,:), 'color', 'k')
      % Field of View
      plot(FOV(:,1,j),FOV(:,2,j), 'b');
      % Target
      sct(j) = scatter(ax, ground_truth(1),ground_truth(2), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g');
      line(target_loc(1,:), target_loc(2,:), 'color', 'g')
      % Observation
      [obs_in, obs_on] = inpolygon(obs(1,:),obs(2,:), FOV(:,1,j),FOV(:,2,j));
      obs_in_on = obs_in | obs_on;
      scatter(ax, obs(1,obs_in_on),obs(2,obs_in_on), 5, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'marker', '+');
      scatter(ax, obs(1,~obs_in_on),obs(2,~obs_in_on), 5, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');

      hold off;
      xlim([0 500])
      ylim([0 500])
      title(['ICF at C_'  num2str(j)]);
    end
    F(i) = getframe(gcf) ;
    drawnow
    pause(0.002)
end

% create the video writer with 1 fps
writerObj = VideoWriter('myVideo.avi');
writerObj.FrameRate = 10;
% set the seconds per image
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(F)
  % convert the image to a frame
  frame = F(i) ;
  writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);
