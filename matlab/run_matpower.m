define_constants;
mpc = loadcase('case118');
results = runpf(mpc);
states = [];
for i=1:10
    states = add_state(results, states);
end
writematrix(states, "118_state.csv");

function states = add_state(results, states)
    define_constants;
    state = ["Date" "Time"];
    for i=1:length(results.bus(:, VM))
        gen_idx = find(results.gen(:, GEN_BUS) == i);
        if ~isempty(gen_idx)
            state(end + 1) = results.gen(gen_idx, QG);
            state(end + 1) = results.gen(gen_idx, PG);
        else
            state(end + 1) = 0;
            state(end + 1) = 0;
        end
        state(end + 1) = results.bus(i, QD);
        state(end + 1) = results.bus(i, PD);
        state(end + 1) = results.bus(i, VA);
        state(end + 1) = results.bus(i, VM) * results.bus(i, BASE_KV);
    end
    state(end + 1) = "no";
    states = [states;state];

end