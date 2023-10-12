P = 'Data/matlab_file';
S = dir(fullfile(P,'*.mat'));
for k = 1:numel(S)
    F = fullfile(S(k).folder,S(k).name);
    D = load(F);
    full_data = [D.absCoMX, D.absCoMY, D.onLED]
    output_path = sprintf("./Data/csv/%1$s.csv",S(k).name)
    csvwrite(output_path, full_data)
    %... your code that processes the imported file data, accessing the fields of D
    %... for example: D.whateverFieldYOuNeed
end

