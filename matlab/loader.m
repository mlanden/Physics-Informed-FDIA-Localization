swat = readtable("C:\Users\mland\OneDrive - Georgia Institute of Technology\Documents\Datasets\SWAT\SWaT_Dataset_Normal_v0.csv");
experiment = getExperiment(swat, 4, 2, [2 3], 3);

%fit101=table2array(swat(80000:end, 2));
%lit101=table2array(swat(80000:end, 3));
%mv101=table2array(swat(80000:end, 4));

%fit101 = fit101 - mean(fit101);
%lit101 = lit101 - mean(lit101);

%z = iddata(mv101, [fit101 lit101], 1);

%z.InputName = ["Flow meter", "Raw tank level"];
%z.OutputName = "Raw tank valve";

%size = 7200*4;
%ze = z(1:size);
%zv = z(size:2 * size);

%delayest(ze)
%delays=ans;
%nn1=struc(1:5,1:5,0);
%nn2=struc(1:5,1:5,2);
%selstruc(arxstruc(ze(:,:,1), zv(:,:,1), nn1))
%selstruc(arxstruc(ze(:,:,2),zv(:,:,2),nn2))
%marx = arx(ze, "na", 4, "nb", [2 1], "nk", [0 2]);
%mn4sid = n4sid(ze, 2:8, "InputDelay", [0 2]);
function experiment = getExperiment(data, index, target, inputs, outputs)
    start = 1;
    experiment = 0;
    i = 0;
    idx_data = table2array(data(:,index));
    while start < length(idx_data) && i < 1000
        i = i + 1;
        start = find(idx_data(start:end)==target, 1);
        after = find(idx_data(start + 1:end)~=target, 1) - 1;
       
        input_values = table2array(data(start:after,inputs));
        output_values = table2array(data(start:after,outputs));
        exp = iddata(output_values, input_values, 1);
        if ~isa(experiment, "iddata")
            experiment = exp;
        else
            experiment = [experiment; exp];
        end
        start = after + 1;
    end
end 