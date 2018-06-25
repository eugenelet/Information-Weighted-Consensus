classdef ICF_node
  properties
    x  % state estimate
    V  % information matrix
    v  % information vector
    H  % observation matrix
    eps  % consensus rate parameter
    K  % consensus iteration
    phi  % state transition matrix
    Q  % process covariance
    W  % prior information matrix
    adj_V  % adjacent node's V
    adj_v  % adjacent node's v
  end
  methods
    function obj = ICF_node(x, W, H, eps, K, phi, Q)
      if nargin > 0
        obj.x = x;
        obj.W = W;
        obj.H = H;
        obj.eps = eps;
        obj.K = K;
        obj.phi = phi;
        obj.Q = Q;
      end
    end
    function obj = compute_information(obj, B, z)
      if nargin > 0
        obj.V = (obj.W / 5) + obj.H'*B*obj.H;
        obj.v = (obj.W*obj.x / 5) + obj.H'*B*z;
      end
    end

    function obj = average_consensus(obj)
      V_delta = 0;
      v_delta = 0;
      for i = 1:size(obj.adj_V, 2)
        V_delta = V_delta + (obj.adj_V{i} - obj.V);
        v_delta = v_delta + (obj.adj_v{i} - obj.v);
      end
      obj.V = obj.V + obj.eps * V_delta;
      obj.v = obj.v + obj.eps * v_delta;
    end

    function [obj, W, x] = compute_posterior_state_and_info(obj)
      obj.x = inv(obj.V) * obj.v;
      obj.W = 5 * obj.V;
      x = obj.x;
      W = obj.W;
    end

    function obj = predict_next_step(obj)
      obj.W = inv(obj.phi * inv(obj.W) * obj.phi' + obj.Q);
      obj.x = obj.phi * obj.x;
      % Invert velocity at boundaries
      if(obj.x(1)<=0)
        obj.x(3) = abs(obj.x(3));
      end
      if(obj.x(1)>=500)
        obj.x(3) = -abs(obj.x(3));
      end
      if(obj.x(2)<=0)
        obj.x(4) = abs(obj.x(4));
      end
      if (obj.x(2)>=500)
        obj.x(4) = -abs(obj.x(4));
      end
    end
  end
end
