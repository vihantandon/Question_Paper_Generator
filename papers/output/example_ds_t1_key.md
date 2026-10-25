# Answer key (draft) -- Data Structures T1

Written by the model and checked automatically: check it before use.

Marks per CO: CO2 4, CO3 5, CO4 11; not tested: CO1, CO5

## Q1 (4 marks, DS_01 Linear Data Structures (Array, Linked List, Stack, Queue), CO2 Apply, easy; generated)

- After PUSH(3): A[1]=3(T), A[2]–, A[3]–, A[4]–, A[5]–
After PUSH(7): A[1]=3, A[2]=7(T), A[3]–, A[4]–, A[5]–
After PUSH(2): A[1]=3, A[2]=7, A[3]=2(T), A[4]–, A[5]–
After POP(): remove 2 → A[1]=3, A[2]=7(T), A[3]–, A[4]–, A[5]–
After PUSH(9): A[1]=3, A[2]=7, A[3]=9(T), A[4]–, A[5]–
After POP(): remove 9 → A[1]=3, A[2]=7(T), A[3]–, A[4]–, A[5]–

## Q2 (5 marks, DS_02 Searching and Sorting, CO3 Analyze, medium; generated)

- a) B = [0, 1, 2, 2, 3, 3, 4, 5]

b) After line 8, C holds cumulative counts: C[0]=1, C[1]=2, C[2]=4, C[3]=6, C[4]=7, C[5]=8

c) Counting sort performs a fixed number of passes over the input: one to count frequencies (O(n)), one to compute cumulative counts (O(k)), and one to place elements into B (O(n)). Hence total time is O(n + k).
- note: the paper would print Analyze (CO3's level), but the question itself reads as Apply

Facts from the model's own knowledge (check these):
- Counting sort runs in O(n + k) time
- Counting sort performs a fixed number of passes over the input: one to count frequencies (O(n)), one to compute cumulative counts (O(k)), and one to place elements into B (O(n)). Hence total time is O(n + k).

## Q3 (5 marks, DS_03 Non-Linear DS: Multi List, Tree, Priority Queue (Heaps), CO4 Apply, medium; generated)

- Start with the given array representing a valid max‑heap.
- Insert 45 at the next free position (index 8).
- Percolate up: compare with parent at index 4 (value 20), swap; now at index 4.
- Compare with new parent at index 2 (value 35), swap; now at index 2.
- Compare with parent at index 1 (value 40), swap; now at index 1.
- Final array after insertion: <45, 40, 30, 35, 25, 15, 10, 20>.
- For extract‑max, remove root (45) and replace it with last element (20) at index 1.
- Heapify‑down: compare 20 with children at indices 2 (40) and 3 (30); swap with larger child 40.
- Now 20 is at index 2; compare with its children at indices 4 (35) and 5 (25); swap with larger child 35.
- Now 20 is at index 4 with no larger children, heap property restored.
- Final array after extract‑max: <40, 35, 30, 20, 25, 15, 10>.

## Q4 (6 marks, DS_04 Non-Linear DS: BST, AVL, RB Tree, B/B+ Tree, CO4 Apply, hard; generated)

- Part a: scanf("%d", &key); placed inside a for loop for i = 0 to 8 (9 iterations). Loop header: for(int i=0;i<9;i++) { scanf("%d", &key); root = avlInsert(root, key); }
- Part b: After inserting the keys the AVL tree is balanced with height 4. The printf statement: printf("%d\n", root->height); (or printf("%d\n", height); if height stored separately).
- Both parts use int data type for keys and height, the loop uses int i, and the conditional inside avlInsert (not shown) relies on comparisons of int keys.

Facts from the model's own knowledge (check these):
- A for loop is an appropriate control structure to iterate through a fixed sequence of inputs.
