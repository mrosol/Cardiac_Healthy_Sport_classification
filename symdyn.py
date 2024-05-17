import numpy as np

def max_min_method(rr_intervals, quantization_level=6):
    min_val, max_val = np.min(rr_intervals), np.max(rr_intervals)
    thresholds = np.linspace(min_val, max_val, quantization_level + 1)[1:-1]
    symbols = np.digitize(rr_intervals, thresholds)
    return symbols

def sigma_method(rr_intervals, sigma_rate=0.05):
    mu = np.mean(rr_intervals)  # Calculate the mean (μ) of RR intervals
    # Transform RR intervals into symbols based on the given thresholds
    symbols = np.zeros(rr_intervals.shape, dtype=int)

    # Conditions for assigning symbols
    symbols[rr_intervals > (1 + sigma_rate) * mu] = 0
    symbols[(mu < rr_intervals) & (rr_intervals <= (1 + sigma_rate) * mu)] = 1
    symbols[(rr_intervals <= mu) & (rr_intervals > (1 - sigma_rate) * mu)] = 2
    symbols[rr_intervals <= (1 - sigma_rate) * mu] = 3

    return symbols


def equal_probability_method(rr_intervals, quantization_level=4):
    percentiles = np.linspace(0, 100, quantization_level+1)
    # Find the values at those percentiles in the RR interval data
    percentile_values = np.percentile(rr_intervals, percentiles)
    # Digitize the RR intervals according to the percentile values
    # np.digitize bins values into the rightmost bin, so we subtract 1 to correct this
    symbols = np.digitize(rr_intervals, percentile_values, right=False) - 1
    # Ensure all symbols are within the range 0 to quantization_level-1
    symbols[symbols == -1] = 0
    symbols[symbols == quantization_level] = quantization_level - 1
    return symbols

def form_words(symbols):
    words = [symbols[i:i+3] for i in range(len(symbols) - 2)]
    return words

def classify_and_count(words):
    families = {'0V': 0, '1V': 0, '2LV': 0, '2UV': 0}
    for word in words:
        unique_elements = len(set(word))
        if unique_elements == 1:
            families['0V'] += 1
        elif unique_elements == 2:
            families['1V'] += 1
        elif unique_elements == 3:
            if (word[1] > word[0] and word[2] > word[1]) or (word[1] < word[0] and word[2] < word[1]):
                families['2LV'] += 1
            else:
                families['2UV'] += 1

    for key in families.keys():
        families[key] = families[key]/len(words)
    return families

# EXAMPLE USAGE
# symbols = equal_probability_method(rr_intervals,4)
# words = form_words(symbols)
# families = classify_and_count(words)