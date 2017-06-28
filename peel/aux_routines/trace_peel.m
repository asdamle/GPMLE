function t = trace_peel(G_D)
    % Estimate trace
    t = 0;
    for jBox = 1:length(G_D)
        t = t + trace(G_D{jBox,1});
    end
end