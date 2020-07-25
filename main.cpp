#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <unordered_map>
#include <cmath>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <numeric>

using namespace std;

class Node {
public:
    Node(): last_char(false), child(vector<Node*>(26, nullptr)) {}
    bool last_char;
    vector<Node*> child;
};

class DoublyNode {
public:
    int val;
    DoublyNode* prev;
    DoublyNode* next;
    DoublyNode* child;
};

class Trie {
public:
    Trie(int n, int m): vis(vector<vector<bool>>(n, vector<bool>(m, false))), root(new Node()) {
    }
    ~Trie() {
        del_tree(root);
    }
    void insert(const string &s) {
        Node* cur_node = root;
        for (int i = 0; i < s.size(); ++i) {
            if (cur_node->child[s[i] - 'a'] == nullptr) {
                cur_node->child[s[i] - 'a'] = new Node();
            }
            cur_node = cur_node->child[s[i] - 'a'];
            if (i == s.size() - 1) cur_node->last_char = true;
        }
    }
    void grid_search(const vector<vector<char>>& board, int i, int j, vector<string>& ret) {
        if (root->child[board[i][j] - 'a'] == nullptr) return ;
        Node* cur_node = root->child[board[i][j] - 'a'];
        string cur_str;
        dfs(board, i, j, ret, cur_node, cur_str);
    }
private:
    void dfs(const vector<vector<char>>& board, int i, int j, vector<string>& ret, Node* cur_node,
             string& cur_str) {
        vis[i][j] = true;
        cur_str.push_back(board[i][j]);
        if (cur_node->last_char) {
            ret.push_back(cur_str);
            cur_node->last_char = false;
        }
        for (int k = 0; k < 4; ++k) {
            int new_i = i + di[k], new_j = j + dj[k];
            if (new_i >= 0 && new_i < board.size() && new_j >= 0 && new_j < board[0].size()
                && !vis[new_i][new_j]
                && cur_node->child[board[new_i][new_j] - 'a'] != nullptr) {
                dfs(board, new_i, new_j, ret, cur_node->child[board[new_i][new_j] - 'a'],
                    cur_str);
            }
        }
        vis[i][j] = false;
        cur_str.pop_back();
    }
    void del_tree(Node* node) {
        for (int i = 0; i < 26; ++i) {
            if (node->child[i] != nullptr) del_tree(node->child[i]);
        }
        delete node;
    }
    Node* root;
    vector<vector<bool>> vis;
    int di[4] = {0, 1, 0, -1}, dj[4] = {1, 0, -1, 0};
};


struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct LinkedList{
    ListNode* head;
    LinkedList(){
        head = NULL;
    }

    void print()
    {
        struct ListNode* temp = head;
        while (temp != NULL) {
            cout << temp->val << " ";
            temp = temp->next;
        }
    }

    void push(int data)
    {
        ListNode* temp = new ListNode(data);
        temp->next = head;
        head = temp;
    }
};

void print_vector(vector<int> v){
    cout << "[";
    for(int i = 0; i<v.size(); i++){
        cout << v[i] << ", ";
    }
    cout << "]"<<endl;
}

typedef pair<int, int> P;

class Solution{
public:
    int maxDepth(TreeNode* root){
        if(root == NULL) return 0;
        int leftHeight = maxDepth(root -> left);
        int rightHeight = maxDepth(root -> right);
        int height = max(leftHeight, rightHeight) + 1;
        return height;
    }

    int getDepth(TreeNode *root){
        if(!root) return 0;
        return max(getDepth(root -> left), getDepth(root -> right)) + 1;
    }

    void printUtil(TreeNode *root, vector<vector<string>> &ans, int start, int end, int depth){
        if(!root) return;
        int mid = start + (end - start)/2;
        ans[depth][mid] = to_string(root -> val);
        printUtil(root -> left, ans, start, mid - 1, depth + 1);
        printUtil(root -> right, ans, mid + 1, end, depth + 1);
    }

    vector<vector<string>> printTree(TreeNode* root){
        int d = getDepth(root);
        vector<vector<string>> ans(d, vector<string>(pow(2, d) - 1, ""));
        printUtil(root, ans, 0, pow(2, d) - 1, 0);
        return ans;
    }

    bool isValidBST(TreeNode* root, long min = LONG_MIN, long max = LONG_MAX){
        if(!root) return true;
        if(root -> val <= min || root -> val >= max) return false;
        return isValidBST(root -> left, min, root -> val) && isValidBST(root -> right, root -> val, max);
    }

    int numTrees(int n){
        int BST[n + 2];
        BST[0] = 1;
        BST[1] = 1;

        for(int i = 2; i <= n; i++){
            BST[i] = 0;
            for(int j = 0; j < i; j++){
                BST[i] += BST[j] * BST[i - j - 1];
            }
        }return BST[n];
    }

    int searchInsert(vector<int>& nums, int target){
        if(target > nums.at(nums.size() - 1)) return nums.size();
        int l = 0;
        int r = nums.size() - 1;

        while(l < r){
            int m = l + (r - l)/2;
            if(target > nums.at(m)){
                l = m + 1;
            }else{
                r = m;
            }
        }
        return l;
    }
    //sum root to leaf leetcode challenge
    int helper(TreeNode* root, int sum){
        if(root == NULL) return 0;
        sum = sum*10 + root -> val;
        if(root -> left == NULL && root -> right == NULL){
            return sum;
        }
        return helper(root -> left, sum) + helper(root -> right, sum);
    }
    int sumNumbers(TreeNode* root) {
        return helper(root, 0);
    }

    //lowest common ancestor BST
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q){
        if(p -> val < root -> val && q -> val < root -> val) return lowestCommonAncestor(root -> left, p, q);
        if(p -> val > root -> val && q -> val > root -> val) return lowestCommonAncestor(root -> right, p, q);
        return root;
    }

//End of tree questions

//Reverse linkedlist
    ListNode* reverseList(ListNode* head) {
        ListNode* current = head;
        ListNode *prev = NULL, *next = NULL;
        while(current != NULL){
            next = current -> next;
            current -> next = prev;
            prev = current;
            current = next;
        }
        head = prev;
        return head;
    }

    void moveZeroes(vector<int>& nums){
        int count = 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums.at(i) != 0){
                nums.at(count++) = nums.at(i);
            }
            while(count > nums.size()){
                nums.at(count++) = 0;
            }
        }
    }

    int uniquePaths(int m, int n){
        int dp[m][n];
        for(int i = 0; i < m; i++){
            dp[i][0] = 1;
        }
        for(int i = 0; i < n; i++){
            dp[0][i] = 1;
        }

        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    void insertionSort(int arr[], int n){
        int i, key, j;
        for(i = 1; i < n; i++){
            key = arr[i];
            j = i - 1;

            while(j >= 0 && arr[j] > key){
                arr[j + 1] = arr[j];
                j--;
                arr[j + 1] = key;
            }
        }
    }

    //coin change dp problem
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1);
        dp.at(0) = 1;

        for(int coin: coins){
            for(int i = coin; i <= amount; i++){
                dp.at(i) += dp.at(i - coin);
            }
        }return dp.at(amount);
    }

    //reconstruct itinerary graph problem
    void dfs(string src,unordered_map<string,vector<string>>& m,vector<string>& ans){
        while(!m[src].empty()){
            string s=m[src].back();
            m[src].pop_back();
            dfs(s,m,ans);
        }
        ans.push_back(src);
    }

    vector<string> findItinerary(vector<vector<string>>& tickets) {
        unordered_map<string,vector<string>> m;
        // Create Graph
        for(auto i:tickets)
            m[i[0]].push_back(i[1]);
        // Sorting in descending order since we will be popping elements from the end
        for(auto &i:m)
            sort(i.second.begin(),i.second.end(),greater<string>());

        vector<string> ans;
        dfs("JFK",m,ans);
        reverse(ans.begin(),ans.end());
        return ans;
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        vector<string> ret;
        if (board.size() == 0 || board[0].size() == 0 || words.size() == 0) return ret;
        int n = board.size(), m = board[0].size();
        Trie trie(n, m);
        for (auto& word: words) {
            trie.insert(word);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                trie.grid_search(board, i, j, ret);
            }
        }
        return ret;
    }

    int arrangeCoins(int n) {
        int count = 0;
        int i = 1;
        while(n){
            n -= i;
            if(n < 0) break;
            count++;
            i++;
        }return count;
    }

    int longestCommonSubsequence(string text1, string text2){
        int m = text1.size();
        int n = text2.size();
        int dp[m + 1][n + 1];
        for (int i = 0; i <= m; i++)
        {
            for (int j = 0; j <= n; j++)
            {
                if (i == 0 || j == 0)
                    dp[i][j] = 0;

                else if (text1[i - 1] == text2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1] + 1;

                else
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }

    vector<int> prisonAfterNDays(vector<int>& cells, int N) {
        map <int, vector <int> > m;
        if(N == 0) return cells;
        set <vector <int> > visited;
        visited.insert(cells);
        for(int i = 1; i<=14 ; i++ ){
            vector <int> temp(8);
            for(int j = 1; j < 7; j++){
                if(cells[j - 1] ^ cells[j + 1] == 0){
                    temp[j] = 1;
                }
            }
            cells = temp;
            m[i] = temp;
            visited.insert(temp);
        }
        return m[N % 14 == 0? 14 : N % 14];
    }

    int nthUglyNumber(int n) {
        int ugly[n]; // To store ugly numbers
        int i2 = 0, i3 = 0, i5 = 0;
        int next_multiple_of_2 = 2;
        int next_multiple_of_3 = 3;
        int next_multiple_of_5 = 5;
        int next_ugly_no = 1;

        ugly[0] = 1;
        for (int i=1; i<n; i++)
        {
            next_ugly_no = min(next_multiple_of_2,
                               min(next_multiple_of_3,
                                   next_multiple_of_5));
            ugly[i] = next_ugly_no;
            if (next_ugly_no == next_multiple_of_2)
            {
                i2 = i2+1;
                next_multiple_of_2 = ugly[i2]*2;
            }
            if (next_ugly_no == next_multiple_of_3)
            {
                i3 = i3+1;
                next_multiple_of_3 = ugly[i3]*3;
            }
            if (next_ugly_no == next_multiple_of_5)
            {
                i5 = i5+1;
                next_multiple_of_5 = ugly[i5]*5;
            }
        } /*End of for loop (i=1; i<n; i++) */
        return next_ugly_no;
    }

    int hammingDistance(int x, int y){
        int count = 0;
        while(x > 0 || y > 0){
            count += (x % 2)^(y % 2);
            x >>= 1;
            y >>= 1;
        }return count;
    }

    vector<vector<int>> threeSum(vector<int>& nums) {
        if(nums.size() < 3) return {};
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < nums.size()-2; i++){
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            else if(nums[i] > 0) break;
            int j = i + 1, k = nums.size() - 1;
            while(j < k){
                if (nums[j]+nums[k]==-nums[i]){
                    res.push_back({nums[i], nums[j], nums[k]});
                    while (j+1<nums.size() && nums[j+1]==nums[j]) j++;
                    j++;
                }else if (nums[j]+nums[k]<-nums[i]){
                    j++;
                }else{
                    k--;
                }
            }
        }
        return res;
    }

    int widthOfBinaryTree(TreeNode* root) {
        int ans = 0;

        queue<pair<TreeNode*, unsigned long long>> q;
        if (root)
            q.push({root, 1});

        while (!q.empty()) {
            int cnt = q.size();
            unsigned long long left = q.front().second, right;
            for (int i = 0; i < cnt; i++) {
                TreeNode* n = q.front().first;
                right = q.front().second;
                q.pop();
                if (n->left != nullptr) {
                    q.push({n->left, 2*right});
                }
                if (n->right != nullptr) {
                    q.push({n->right, 2*right+1});
                }
            }
            //cout << right << " " << left << "\n";
            ans = max(ans, (int)(right - left + 1));
        }

        return ans;
    }

    DoublyNode* flatten(DoublyNode* head) {
        if(!head) return NULL;
        DoublyNode* trav = head;
        while(trav){
            if(trav->child){
                DoublyNode* next = trav->next;
                DoublyNode* child = flatten(trav->child);
                trav->child = NULL;
                trav->next = child;
                child->prev = trav;
                DoublyNode* lastNode = child;
                while(lastNode->next) lastNode = lastNode->next;
                lastNode->next = next;
                if(next) next->prev = lastNode;
            }
            trav = trav->next;
        }
        return head;
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector< vector<int>> subset;
        vector<int> empty;
        subset.push_back(empty);

        for (int i = 0; i < nums.size(); i++)
        {
            vector< vector<int> > subsetTemp = subset;  //making a copy of given 2-d vector.

            for (int j = 0; j < subsetTemp.size(); j++)
                subsetTemp[j].push_back( nums[i] );   // adding set[i] element to each subset of subsetTemp. like adding {2}(in 2nd iteration  to {{},{1}} which gives {{2},{1,2}}.

            for (int j = 0; j < subsetTemp.size(); j++)
                subset.push_back( subsetTemp[j] );  //now adding modified subsetTemp to original subset (before{{},{1}} , after{{},{1},{2},{1,2}})
        }
        return subset;
    }

    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        int b = 32;
        while(b--){
            res <<= 1;
            res += n % 2;
            n >>= 1;
        }
        return res;
    }

     bool isSameTree(TreeNode* p, TreeNode* q) {
        if(p == NULL && q == NULL) return true;
        if(p == NULL || q == NULL) return false;
        if(p->val == q->val) return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        return false;
    }

    double angleClock(int hour, int minutes) {
        // Degree covered by hour hand (hour area + minutes area)
        double h = (hour%12 * 30) + ((double)minutes/60 * 30);

        // Degree covered by minute hand (Each minute = 6 degree)
        double m = minutes * 6;

        // Absolute angle between them
        double angle = abs(m - h);

        // If the angle is obtuse (>180), convert it to acute (0<=x<=180)
        if (angle > 180) angle = 360.0 - angle;

        return angle;
    }

    string reverseWords(string s) {
        reverse(s.begin(), s.end());

        //reverse words
        for(int i = 0, j = 0;i<s.size();){
            while(i<s.size() && (i < j || s[i]==' ') ) i++;
            while(j<s.size() && (j<i || s[j]!=' ')) j++;
            reverse(s.begin()+i, s.begin()+j);
        }

        //remove space
        int i = 0,j = 0;
        while(j<s.size()){
            if(s[j]!=' ' || (s[j]==' ' && i!=0 && j!= s.size()-1 && s[j+1]!= ' '))
                s[i++] = s[j++];
            else if(s[j]==' ')
                j++;
        }
        s.resize(i);
        return s;
    }

public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> cnt;
        for (int x : nums) 	cnt[x] ++;
        priority_queue<P, vector<P>, greater<P> > q;
        for (auto &x : cnt) {
            if (q.size() < k)
                q.push(make_pair(x.second, x.first));
            else {
                if (q.top().first < x.second) {
                    q.pop();
                    q.push(make_pair(x.second, x.first));
                }
            }
        }
        vector<int> ans;
        while (!q.empty()) {
            ans.push_back(q.top().second);
            q.pop();
        }
        return ans;
    }

    string addBinary(string a, string b) {
        if (a.size() < b.size()) {
            string tmp = a;
            a = b;
            b = tmp;
        }

        reverse(a.begin(), a.end());
        reverse(b.begin(), b.end());

        for (int i = 0; i < b.size(); i++) {
            int na = a[i] - '0';
            int nb = b[i] - '0';

            if (na + nb > 1) {
                a[i] = ('0' + na + nb - 2);
                if (i == a.size() - 1) a += "1";
                else a[i + 1]++;
            }
            else {
                a[i] = ('0' + na + nb);
            }
        }

        for (int i = b.size(); i < a.size(); i++) {
            if (a[i] == '2') {
                a[i] = '0';
                if (i == a.size() - 1) a += "1";
                else a[i + 1]++;
            }
            else break;
        }

        reverse(a.begin(), a.end());
        return a;
    }

private:
    int m, n;
    bool search_at(vector<vector<char>>& board, const string word, int char_idx, int x, int y){
        char curr = board[x][y];
        bool result = false;
        if (curr != word[char_idx]) return result;
        if (word.size() == char_idx+1) return true;

        board[x][y] = '-';
        if (x>0 && search_at(board, word, char_idx+1, x-1, y))  result = true;
        if (!result && y>0 && search_at(board, word, char_idx+1, x, y-1))  result = true;
        if (!result && y<n-1 && search_at(board, word, char_idx+1, x, y+1))  result = true;
        if (!result && x<m-1 && search_at(board, word, char_idx+1, x+1, y))  result = true;

        board[x][y] = curr;
        return result;
    }
public:
    bool exist(vector<vector<char>>& board, string word) {
        m = board.size(), n = board[0].size();
        if (word.size() > m * n) return false;

        for (int idx1 = 0; idx1 < m; idx1++){
            for (int idx2 = 0; idx2 < n; idx2++){
                if (search_at(board, word, 0, idx1, idx2)) return true;
            }
        }
        return false;
    }

public:
    void fill(TreeNode * root,vector<vector<int>> & ans,int level)
    {
        if(!root)return;     // if null return
        if(ans.size() <= level) // maintains appropriate array size, if not you'll get out of bounds exception
            ans.push_back({});
       fill(root -> left,ans,level+1); // always increment level by one as we go down the tree
        if(level%2 == 0) // if its even, a simple inorder traversal is fine
            ans[level].push_back(root -> val);
        else
            ans[level].insert(ans[level].begin(),root -> val);  // else insert the value into the front of our level
       fill(root -> right,ans,level+1);
    }
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
       vector<vector<int>> ans;
       fill(root,ans,0); // level starts at 0
       return ans;
    }

    vector<int> singleNumber(vector<int>& nums) {
        int difference = accumulate(nums.begin(), nums.end(), 0, bit_xor<int>());
        int mask = difference & - difference;
        int x = 0, y = 0;
        for(int num: nums){
            if(num & mask) x ^= num;
            else y^= num;
        }
        return {x,y};
    }

    vector<vector <int>> res;
    void solve(vector < vector <int> >& graph, int node, int target, vector <int>temp){
      temp.push_back(node);
      if(node == target){
         res.push_back(temp);
         return;
      }
      for(int i = 0; i < graph[node].size(); i++){
         solve(graph, graph[node][i], target, temp);
      }
   }
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        vector<int> temp;
        solve(graph, 0, graph.size() - 1, temp);
        return res;
    }
    int findMin(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        return nums.at(0);
    }
};

class graph {
    int V; //number of vertices
    list<int> *adjList; //adjacency list
    void DFS_helper(int v, bool visited[]);
public:
    //constructor
    graph(int V){
        this -> V = V; //this pointer referring to the current object
        adjList = new list<int>[V];
    }

    //adding an edge to the graph
    void addEdge(int v, int w){
        adjList[v].push_back(w);
    }
    void DFS(); //calling the traversal function
    void BFS(int key);
};

void graph::DFS_helper(int v, bool visited[]){
    //current node v is visited
    visited[v] = true;
    cout << v << " ";
    //recursively process all the adjacent vertices of the node
    list<int>::iterator i;
    for(i = adjList[v].begin(); i != adjList[v].end(); i++){
        if(! visited[*i])
            DFS_helper(*i, visited);
    }
}

void graph::DFS(){
    //initially none are visited
    bool *visited = new bool[V];
    for(int i = 0; i < V; i++){
        visited[i] = false;
    }
    // explore the vertices one by one by recursively calling DFS_helper
    for(int i = 0; i < V; i++){
        if(!visited[i]) DFS_helper(i, visited);
    }
}

void graph::BFS(int s){
    bool *visited = new bool[V];
    for(int i = 0; i < V; i++){
        visited[i] = false;
    }
    list<int> queue;
    visited[s] = true;
    queue.push_back(s);
    list<int>::iterator i;

    while(!queue.empty()){
        s = queue.front();
        cout << s << " ";
        queue.pop_front();

        for(i = adjList[s].begin(); i != adjList[s].end(); i++){
            if(!visited[*i]){
                visited[*i] = true;
                queue.push_back(*i);
            }
        }
    }
}


int main(){
    graph gdfs(5);
    gdfs.addEdge(0, 1);
    gdfs.addEdge(0, 2);
    gdfs.addEdge(0, 3);
    gdfs.addEdge(1, 2);
    gdfs.addEdge(2, 4);
    gdfs.addEdge(3, 3);
    gdfs.addEdge(4, 4);
    cout << "DFS traversal of the following graph \n";
    gdfs.DFS();
    cout << "\n";
    int vertex = 0;
    cout << "BFS traversal starting from vertex: " << vertex << "\n";
    gdfs.BFS(vertex);

    Solution sol;
    vector<int> zeroes = {1,0,0,2,3,4};
    sol.moveZeroes(zeroes);

    int arr[] = {4,3,6,1,7,2,9};
    int n = 7;
    sol.insertionSort(arr, n);
    for (const auto& e : arr) {
        cout << e << endl;
    }

    //Implement reverse linkedlist
    LinkedList ll;
    ll.push(1);
    ll.push(2);
    ll.push(3);
    ll.push(4);
    ll.push(5);
    cout << "Given linkedlist\n";
    ll.print();
    cout << "\nReversed LinkedList\n";
    ListNode* head = ll.head;
    sol.reverseList(head);
    cout << sol.hammingDistance(12,13);
    return 0;
}