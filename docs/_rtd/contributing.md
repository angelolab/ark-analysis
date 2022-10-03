## Contributing to ark-analysis
Thank you for your interest in contributing to our open source project! Below are some helpful guidelines which describe our workflow. This should help you get up to speed and oriented to our project. 

### Step 1: Identify an issue to work on  
Issues are how we keep track of known bugs, areas that can be improved, and community suggestsions for new features. Before writing any code, you should first identify an issue which describes the problem you'd like to address. Our list of currently open issues is displayed on our [issues page](https://github.com/angelolab/ark-analysis/issues).

If you have your own idea for an improvement or change you'd like to make, you can [create a new issue](https://github.com/angelolab/ark-analysis/issues/new/choose) describing your idea. Issues provide a convenient format to have a discussion about what the specific problem is, as well as the potential ways to address it. 

Either way, whether you decide to tackle an existing issue or open your own, make sure to start a conversation with our team by replying to issue itself so that we can give you all the context you need to understand the problem. 

### Step 2: Create an outline of your proposed solution
Once you've identified the issue that you'd like to work on and have come to a consensus with our developers about what approach you're going to take to tackle it, the next step is creating a [design document](https://github.com/angelolab/ark-analysis/issues/new?assignees=&labels=design_doc&template=design_doc.md&title=). Design docs are a more fine-grained description of what code you're planning on writing. We've found that spending an hour or two putting together a design doc can often save days (or more!) of developer time by identifying misunderstandings before any code has been written. They're also a great way to get feedback from our team to make sure the implementation you're planning will integrate seamlessly with the rest of the codebase. 

Once your design doc is ready to be reviewed, let us know, and we'll provide any feedback we think is necessary. Once everyone is on the same page, you're ready to get started!

### Step 3: Create a local branch to hold your changes. 
In order to keep track of the many different improvements that are being worked on at once, we use different branches in git. This keeps these changes isolated from each other during development. 

If you'd like a refersher on using git and why it's useful, check out [this git reference](https://git-scm.com/book/en/v2). If you'd like an overview on collaborating via github, check out [this tutorial](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests). 

You should create a branch specific to the current design doc you're working on, with an informative, short name. For an overview of how to set up your computing environment for development, see our [developer's guide](https://ark-analysis.readthedocs.io/en/latest/_rtd/development.html#)

### Step 4: Coding time
Now that you're ready to get to work, you can make the necessary modifications to our codebase to address your issue. 

There are a few important details to keep in mind as you're writing your code. The first is that we follow [the google styleguide](https://google.github.io/styleguide/pyguide.html) for python code. It's good to take a look through here if you aren't familiar with it, to get a sense for what we  expect. You can also look through our [source code](https://github.com/angelolab/ark-analysis/tree/main/ark) to see how we've implemented these suggestions.

The second important concept is [modular code](https://opensource.com/article/19/7/get-modular-python-functions). Breaking your code up into small pieces will make it easier to read, easier to understand, and easier to reuse. Before submitting your PR, take a look through your code to see if it could be broken up into smaller, logical pieces that work together, rather than a few large chunks that do everything at once. 

The third important concept is [testing](https://realpython.com/python-testing/). All of the code that gets added to our repository must be tested! This allows us to make sure that it's working as intended. Even more importantly, it means that if someone makes a change in the future that causes the code to break, we'll identify that problem in advance so that it can be fixed. Once you've finished creating your code, it's important to write test functions to ensure that it's behaving as expected. For an example of what that looks like, you can take a look at our [data_utils](https://github.com/angelolab/ark-analysis/blob/main/ark/utils/data_utils.py) and the [associated test code](https://github.com/angelolab/ark-analysis/blob/main/ark/utils/data_utils_test.py). 

### Step 5: Submit your code to be reviewed
Once you think you have a version that's ready for us to look at, you can submit a pull request for us to look at. For more details on how this works, see the [previous link](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests) on using github. 

Pull requests allow us to look at proposed changes, and give feedback prior to integrating them into the main version of the repo that everyone is using. After you open a new pull request, a number of automatic checks will run. For example, you might see an error message from Travis indicating the build failed: 
![image](https://user-images.githubusercontent.com/13770365/91110453-c10f9a80-e632-11ea-831a-785318d1dd94.png)

This means that some of the tests didn't pass. You can click on the link for more information about which tests specifically failed. 

Once all of the tests have passed, you should request a review from someone on our team. Chances are, the person who you were communicating with on the linked issue is the best person to review your PR.

### Step 6: Respond to review comments
No one writes perfect code the first time. Chances are, your reviewer will have some suggested changes for your code. Take the time to carefully read through their comments, and make sure to ask any clarifying questions. Then, once you understand what's being asked of you, update your PR with the requested changes. You can continue to make commits to the existing branch that you used to create your PR. As you push new commits to that branch, the PR will automatically update, and Travis will automatically re-run to test this new code. Once you've addressed the comments made during review, you can click the 're-request review' button on the top. Once your reviewer is satisfied with your changes, they will approve your PR and merge it into the main branch. 

Thanks in advance for contributing to our project!
