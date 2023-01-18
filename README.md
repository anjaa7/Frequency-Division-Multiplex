# Frequency Division Multiplex

FDM is a method for transmitting multiple signals simultaneously over a single channel by dividing the channel into different frequency bands, each carrying a separate signal. An FDM system for independent parallel transmission of two signals would modulate them onto different carrier frequencies and transmit them over the same channel, then demodulate and separate them by their frequency band at the receiving end. This allows for simultaneous transmission of multiple signals without interference.

## Explanation of the each filter in the diagram:
### NF filtersat at the input :
* Filters used to limit the frequency band of the input signals 𝑦1(𝑡) and 𝑦2(𝑡) to specific frequency ranges.
### AM modulator:
* It modulates the amplitude of 𝑦2(𝑡) with a carrier frequency 𝑓𝑐.
### Connection channel:
* The channel through which the modulated signals are transmitted.
### PO filter: 
* It is used to eliminate the unwanted frequency components of the received signal.
### FM demodulator:
* It demodulates the received signal 𝑦𝑅(𝑡) to obtain the original signal 𝑦2(𝑡).
### LF filters at the output:
* These filters are used to limit the frequency band of the received signals 𝑦1𝑟(𝑡) and 𝑦2𝑟(𝑡) to specific frequency ranges.
