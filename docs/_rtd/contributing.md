## Contributing to ark-analysis
Thank you for your interest in contributing to our open source project! Below are some helpful guidelines to keep in mind before you create a new issue or open a pull request

### Creating an issue
Issues are the way that github tracks bugs, requests, or general discussions about functionality within a repository. Before you create an issue, make sure to first take a look [at the currently open issues](https://github.com/angelolab/ark-analysis/issues) to see if there is a relevant ongoing discussion. 

Once you've determined that your issue really is a new issue, fill out the appropriate [issue template](https://github.com/angelolab/ark-analysis/issues/new/choose).

### Creating a pull request

#### Before you start coding
Pull requests (PRs) are how new code gets added to the project. They faciliate code review, which is important to make sure any newly added code meets our quality standards and adds useful features. Before starting a PR, it's a good idea to first open an issue with either a bug report or an enhancement. This will allow discussion about the proposed change before any code has been written, which will save everyone time. 

Once you've decided to start working on an issue, please 'assign' that issue to yourself so that others know you're working on it. This prevents duplicate work and allows us to keep track of who is doing what. 

If you'd like a refersher on using git and why it's useful, check out [this git reference](https://git-scm.com/book/en/v2). If you'd like an overview on collaborating via github, check out [this tutorial](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests)

#### While you're coding
There are a few important details to keep in mind as you're writing your code. The first is that we follow [the google styleguide](https://google.github.io/styleguide/pyguide.html) for python code. It's good to take a look through here if you aren't familiar with it, to get a sense for what we  expect. You can also look through our [source code](https://github.com/angelolab/ark-analysis/tree/master/ark) to see how we've implemented these suggestions

The second important concept is [modular code]. Breaking your code up into small pieces will make it easier to read, easier to understand, and easier to reuse. Before submitting your PR, take a look through your code to see if it could be broken up into smaller, logical pieces that work together, rather than a few large chunks that do everything at once. 

The third important concept is [testing](https://realpython.com/python-testing/). All of the code that gets added to our repository must be tested! This allows us to make sure that it's working as intended. Even more importantly, it means that if someone makes a change in the future that causes the code to break, we'll identify that problem during the pull-request stage, where it can be fixed. Before submitting your PR, make sure that you've written a test function covering all of the new features you've included in your PR. 

#### After you've finished coding
Once you think you have a version that's ready for us to look at, you can submit a pull request for us to look at. After you open a new PR, a number of automatic checks will run. For example, you might see an error message from Travis indicating the build failed: 
![image](https://user-images.githubusercontent.com/13770365/91110453-c10f9a80-e632-11ea-831a-785318d1dd94.png)

This means that some of the tests didn't pass. You can click on the link for more information about which tests specifically failed. 

Once all of the tests have passed, you can request a review. Chances are, the person who you were communicating with on the linked issue is the best person to review your PR.

#### After you've gotten review comments
No one writes perfect code the first time. Chances are, your reviewer will have some suggested changes for your code. Take the time to carefully read through their comments, and make sure to ask any clarifying questions. Then, once you understand what's being asked of you, update your PR with the requested changes. You can continue to make commits to the existing branch that you used to create your PR. As you push new commits to that branch, the PR will automatically update. 

Thanks in advance for contributing to our project!
