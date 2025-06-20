export function createRandomSource() {
    return {
        /**
         * Return an array of length d containing values in [-1,1].
         * @param {number} d
         * @returns {number[]}
         */
        nextVector(d) {
            return Array.from({ length: d }, () => Math.random() * 2 - 1);
        },

        /**
         * Return a Float32Array of random numbers for an attention matrix.
         * @param {number} heads
         * @param {number} tokens
         * @returns {Float32Array}
         */
        nextAttention(heads, tokens) {
            const arr = new Float32Array(heads * tokens * tokens);
            for (let i = 0; i < arr.length; i++) arr[i] = Math.random();
            return arr;
        }
    };
} 