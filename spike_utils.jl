# Utility functions for spike thresholding

using DSP

"""
  load_fly_data(filepath, normalize_stim=true)

Loads the H1 fly data. Return the following:
- spk : Extracellular potential
- stim : Stim trace
- t : Time vector
- srate : Sampling rate
- pts_per_ms : datapoints per millisecond
"""
function load_fly_data(filepath, normalize_stim=true)
    # Read in file
    dat = pyabf.ABF(filepath)
    # Extract all data
    srate = dat.dataRate
    data = dat.data
    spk  = data[1,:] .- mean(data[1,:], dims=1)
    t    = dat.sweepX
    pts_per_ms = dat.dataPointsPerMs
    if normalize_stim
        stim = data[2,:] .- minimum(data[2,:])
        stim = stim ./ maximum(stim)
    else
        stim = data[2,:]
    end
    return spk, stim, t, srate, pts_per_ms
end


function load_crayfish_data(filepath, normalize_stim=true)
    # Read in file
    dat = pyabf.ABF(filepath)
    # Extract all data
    srate = dat.dataRate
    data = dat.data
    spk  = data[1,:] .- mean(data[1,:], dims=1)
    t    = dat.sweepX
    pts_per_ms = dat.dataPointsPerMs
    if normalize_stim
        stim = data[2,:] .- minimum(data[2,:])
        stim = stim ./ maximum(stim)
    else
        stim = data[2,:]
    end
    return spk, stim, t, srate, pts_per_ms
end


"""
Thresholds voltage signal at given threshold and returns indices
of detected spiketimes
"""
function get_thresholded_spikes(signal, threshold, spk_win_pts)
    # Grab all treshold crossings
    if threshold < 0
        thresh_crossings = findall(signal .< threshold)
    else
        thresh_crossings = findall(signal .> threshold)
    end
    pushfirst!(thresh_crossings, 1)
    # Keep only the first timepoint of each threshold crossing, and
    # ensure at least 3.5 milliseconds between detected spikes
    diffs = diff(thresh_crossings)
    spkidx = []
    for i = 1 : length(diffs)
        if diffs[i] > 35
            push!(spkidx, thresh_crossings[i+1])
        end
    end
    # Remove spikes which fall too closely to the beginning or end of
    # the recording
    deleteat!(spkidx, findall(s -> s < spk_win_pts || s > length(signal)-spk_win_pts, spkidx))
    return spkidx
end


"""
Collects waveforms into matrix
"""
function collect_waveforms(spk, spkidx, spk_win_ms, pts_per_ms)
    # Time vector for waveform
    tspk = range(-Int(spk_win_ms/2), Int(spk_win_ms/2), length=spk_win_ms*pts_per_ms)
    # Waveform matrix
    wfs = zeros(length(spkidx), length(tspk))
    for i = 1 : length(spkidx)
        ti = spkidx[i]
        wfs[i,:] = spk[ti-Int(length(tspk)/2) : ti+Int(length(tspk)/2)-1]
    end
    return wfs, tspk
end


"""
Computes features of waveforms
"""
function compute_wf_features(wfs, tspk)
    auc     = sum(wfs, dims=2)
    absauc  = sum(abs.(wfs), dims=2)
    spkauc  = sum(abs.(wfs[:, Int(length(tspk)/2):Int(length(tspk)/2)+10]), dims=2)
    peak    = maximum(wfs, dims=2)
    trough  = minimum(wfs, dims=2)
    pt_dist = peak .- trough;
    feats = Dict()
    feats["auc"] = auc
    feats["absauc"] = absauc
    feats["spkauc"] = spkauc
    feats["peak"]   = peak
    feats["trough"] = trough
    feats["pt_dist"] = pt_dist
    return feats
end


function check_bad_wfs(wfs, tspk)
    troughs  = zeros(size(wfs)[1])
    peaks    = zeros(size(wfs)[1])
    toremove = falses(size(wfs)[1])
    for i = 1 : length(troughs)
        # Grab min value within 1 ms post spike to capture the trough
        peaks[i]   = maximum(wfs[i, Int(length(tspk)/2):Int(length(tspk)/2)+20])
        troughs[i] = minimum(wfs[i, Int(length(tspk)/2):Int(length(tspk)/2)+10])
        # Check whether any values in the baseline are lower, and remove
        # those waveforms
        if minimum(wfs[i, 1:Int(length(tspk)/2)]) < troughs[i]
            toremove[i] = true
        elseif maximum(wfs[i, Int(length(tspk)/2)+20:end]) > peaks[i]/2 
            toremove[i] = true    
        elseif troughs[i] < -1000
            toremove[i] = true
        end
    end
    return toremove, peaks, troughs
end


function summarize_session(session)
    # Pull out all variables
    t = session["t"]
    spk = session["spk"]
    spkidx = session["spkidx"]
    troughs = session["troughs"]
    peaks = session["peaks"]
    THRESHOLD = session["THRESHOLD"]
    stim = session["stim"]
    wfs = session["wfs"]
    toremoveidx = session["toremoveidx"]
    tspk = session["tspk"]
    cycle_idx = session["cycle_idx"]
    starting_val = session["starting_val"]

    # Plot the results for this file
    figure()
    subplot(2,2,1)
    plot(t, spk) ; plot(t[spkidx], troughs, "r.", label="detected spike")
    plot(t, THRESHOLD .+ zeros(length(t)), "k--",   label="threshold")
    xlabel("Time [s]") ; ylabel("μV")
    title("Total threshold crossings detected: " * string(length(spkidx)))

    subplot(2,2,3)
    plot(t, stim, "r")
    scatter(t[cycle_idx], starting_val .+ zeros(length(cycle_idx)), marker="*", color="b")
    xlabel("Time [s]") ; ylabel("AU")
    title("Stimulus drum")

    subplot(2,2,2)
    plot(tspk, transpose(wfs[toremoveidx, :][1:100,:]))
    xlabel("Time [ms]") ; ylabel("μV")
    title("Waveforms to remove, n=" * string(sum(toremoveidx)))

    subplot(2,2,4)
    plot(tspk, transpose(wfs[.~toremoveidx,:][1:100, :]))
    xlabel("Time [ms]") ; ylabel("μV")
    title("Waveforms to remaining, n=" * string(sum(.~toremoveidx)))

    suptitle(session["filepath"])
end


function mark_cycles(session)
    stim = session["stim"]
    # Collect indices of stim timeseries sitting at starting value
    starting_val = mean(stim[1:1000])
    cycles = findall(s -> s < starting_val + 0.01 && s > starting_val - 0.01, stim)
    diffs = diff(cycles)
    # Prune to only keep single index for each revolution
    cycle_idx = []
    found_first_idx = false
    for i = 1 : length(diffs)
        if diffs[i] > 100
            if !found_first_idx
                push!(cycle_idx, cycles[i])
                found_first_idx = true
            end
            push!(cycle_idx, cycles[i+1])
        end
    end
    # Prune the even-numbered points
    deleteat!(cycle_idx, 2:2:length(cycle_idx))
    return cycle_idx, starting_val
end


function collect_cycles(session)
    session["trl"] = Dict()
    cycle_idx = session["cycle_idx"]
    for i = 1 : length(cycle_idx) - 1
        itrial_spkidx = findall(
            s_i -> s_i > cycle_idx[i] && s_i < cycle_idx[i+1], session["spkidx"]
        )
        session["trl"][i] = Dict()
        session["trl"][i]["t"]       = session["t"][cycle_idx[i] : cycle_idx[i+1]]
        session["trl"][i]["spk"]     = session["spk"][cycle_idx[i] : cycle_idx[i+1]]
        session["trl"][i]["stim"]    = session["stim"][cycle_idx[i] : cycle_idx[i+1]]
        session["trl"][i]["spkidx"]  = session["spkidx"][itrial_spkidx] .- cycle_idx[i]
        session["trl"][i]["wfs"]     = session["wfs"][itrial_spkidx,:]
        session["trl"][i]["peaks"]   = session["peaks"][itrial_spkidx]
        session["trl"][i]["troughs"] = session["troughs"][itrial_spkidx]
    end
    session["ntrials"] = length(cycle_idx) - 1
    return session
end


function collect_trials(spk, t, stim, spkidx, wfs, peaks, troughs, trialidx)
    trl = Dict()
    for i = 1 : size(trialidx)[1]
        itrial_spkidx = findall(
            s_i -> s_i > trialidx[i,1] && s_i < trialidx[i,2], spkidx
        )
        trl[i] = Dict()
        trl[i]["t"]       = t[trialidx[i,1] : trialidx[i,2]]
        trl[i]["spk"]     = spk[trialidx[i,1] : trialidx[i,2]]
        trl[i]["stim"]    = stim[trialidx[i,1] : trialidx[i,2]]
        trl[i]["spkidx"]  = spkidx[itrial_spkidx] .- trialidx[i,1]
        trl[i]["wfs"]     = wfs[itrial_spkidx,:]
        trl[i]["peaks"]   = peaks[itrial_spkidx]
        trl[i]["troughs"] = troughs[itrial_spkidx]
    end
    trl["ntrials"] = size(trialidx)[1]
    return trl
end


"""
    smoothed_sig = smooth(x, window_len, window)
taken from https://github.com/JuliaDSP/DSP.jl/issues/112#issuecomment-115212185
"""
function smooth(x, window_len)#, window)
    # w = getfield(DSP.Windows, window)(window_len)
    # w = DSP.Windows.gaussian(window_len, 0.25)
    w = ones(window_len)
    return DSP.filtfilt(w ./ sum(w), [1.0], x)
end


# function compute_trl_frs(session, window)
#     for itrl = 1 : length(session["trl"])
#         # Construct binary spike train to convolve on
#         spktrain = zeros( length(session["trl"][itrl]["spk"]) )
#         spktrain[session["trl"][itrl]["spkidx"]] .= 1

#         # Convolve it with window to get instantaneous firing rate
#         ifr_padded = conv(spktrain, window)

#         # Remove padding from the result
#         correct_length = length(spktrain)
#         difference = length(ifr_padded) - correct_length
#         start_idx = Int(round(1 + difference/2))
#         stop_idx  = Int(round(length(ifr_padded) - difference/2))
#         session["trl"][itrl]["ifr"] = ifr_padded[ start_idx : stop_idx ]
#     end
#     return session
# end


function compute_trl_frs(trl, window, pts_per_ms)
    # Compute inst firing rate for every trial
    for i = 1 : trl["ntrials"]
        # Construct binary spike train to convolve on
        spktrain = zeros( length(trl[i]["spk"]) )
        spktrain[trl[i]["spkidx"]] .= 1

        # Convolve it with window to get instantaneous firing rate
        ifr_padded = conv(spktrain, window)

        # Remove padding from the result
        correct_length = length(spktrain)
        difference = length(ifr_padded) - correct_length
        start_idx = Int(round(1 + difference/2))
        stop_idx  = Int(round(length(ifr_padded) - difference/2))
        trl[i]["ifr"]      = ifr_padded[start_idx : stop_idx]
        trl[i]["spktrain"] = spktrain
    end

    # Compute mean IFR
    lengths = []
    for i = 1 : trl["ntrials"]
        push!(lengths, length(trl[i]["ifr"]))
    end
    minlength = minimum(lengths)
    ifrs      = zeros(trl["ntrials"], minlength)
    for i = 1 : trl["ntrials"]
        ifrs[i,:] = trl[i]["ifr"][1:minlength]
    end
    trl["t_ifr"] = range(0, minlength/pts_per_ms/1000, length=minlength)

    # Compute mean spktrain
    lengths = []
    for i = 1 : trl["ntrials"]
        push!(lengths, length(trl[i]["spktrain"]))
    end
    minlength = minimum(lengths)
    spktrains = zeros(trl["ntrials"], minlength)
    for i = 1 : trl["ntrials"]
        spktrains[i,:] = trl[i]["spktrain"][1:minlength]
    end

    trl["ifr"] = transpose(mean(ifrs, dims=1))
    trl["ifrs"] = transpose(ifrs)
    trl["spktrain"] = transpose(mean(spktrains, dims=1))
    trl["spktrains"] = transpose(spktrains)
    return trl
end


function compute_trl_frs_isi_reciprocal(trl, pts_per_ms)
    # Compute inst firing rate for every trial
    for i = 1 : trl["ntrials"]
        # Compute 1 / ISI
        trl[i]["spktimes"] = trl[i]["t"][ trl[i]["spkidx"] ]
        trl[i]["ifr"] = diff(trl[i]["spktimes"])
    end

    # Compute mean IFR
    lengths = []
    for i = 1 : trl["ntrials"]
        push!(lengths, length(trl[i]["ifr"]))
    end
    minlength = minimum(lengths)
    ifrs = zeros(trl["ntrials"], minlength)
    for i = 1 : trl["ntrials"]
        ifrs[i,:] = trl[i]["ifr"][1:minlength]
    end
    trl["t_ifr"] = range(0, minlength/pts_per_ms/1000, length=minlength)

    # Compute mean spktrain
    trl["ifr"] = transpose(mean(ifrs, dims=1))
    trl["ifrs"] = transpose(ifrs)
    return trl
end




