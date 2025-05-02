/**
 * @param {string} s
 * @return {number}
 */
var lengthOfLongestSubstring = function(s) {
    let seen = new Set(); // To store unique characters in the current window
    let left = 0;         // Start of the sliding window
    let maxLength = 0;

    for (let right = 0; right < s.length; right++) {
        // If character already seen, shrink the window from the left
        while (seen.has(s[right])) {
            seen.delete(s[left]);
            left++;
        }

        // Add current character to the set
        seen.add(s[right]);

        // Update maxLength
        maxLength = Math.max(maxLength, right - left + 1);
    }

    return maxLength;
};
