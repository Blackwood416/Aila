#include "ChatTest.hpp"
#include "chat/ThinkingBudgetController.hpp"

namespace aila::chat::test {

void run_thinking_budget_controller_tests() {
    {
        ThinkingBudgetController c;
        c.start(true, 2, 10, 11);
        AILA_CHAT_EXPECT_TRUE(c.enabled());
        AILA_CHAT_EXPECT_TRUE(c.should_disable_chunked_decode());
        AILA_CHAT_EXPECT_TRUE(c.inside_think());
        c.observe_generated_token(100);
        AILA_CHAT_EXPECT_TRUE(!c.needs_forced_close());
        c.observe_generated_token(101);
        AILA_CHAT_EXPECT_EQ(c.used_tokens(), 2);
        AILA_CHAT_EXPECT_TRUE(c.needs_forced_close());
        c.mark_forced_close();
        AILA_CHAT_EXPECT_TRUE(c.forced_close());
        AILA_CHAT_EXPECT_TRUE(!c.inside_think());
    }

    {
        ThinkingBudgetController disabled;
        disabled.start(true, -1, 10, 11);
        AILA_CHAT_EXPECT_TRUE(!disabled.enabled());
        AILA_CHAT_EXPECT_TRUE(!disabled.should_disable_chunked_decode());
    }

    {
        ThinkingBudgetController natural_close;
        natural_close.start(true, 2, 10, 11);
        natural_close.observe_generated_token(100);
        natural_close.observe_generated_token(11);
        natural_close.observe_generated_token(101);
        AILA_CHAT_EXPECT_EQ(natural_close.used_tokens(), 1);
        AILA_CHAT_EXPECT_TRUE(!natural_close.needs_forced_close());
        AILA_CHAT_EXPECT_TRUE(!natural_close.inside_think());
    }

    {
        ThinkingBudgetController opened_by_model;
        opened_by_model.start(false, 1, 10, 11);
        opened_by_model.observe_generated_token(100);
        AILA_CHAT_EXPECT_TRUE(!opened_by_model.needs_forced_close());
        opened_by_model.observe_generated_token(10);
        opened_by_model.observe_generated_token(101);
        AILA_CHAT_EXPECT_TRUE(opened_by_model.needs_forced_close());
    }
}

} // namespace aila::chat::test

void run_thinking_budget_controller_tests() {
    aila::chat::test::run_thinking_budget_controller_tests();
}
